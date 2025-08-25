import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return x

class ChordEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # add positional encoding

        B, T = input_ids.size()  # input: (B, T)

        if attention_mask is not None:
            # Convert attention_mask (1=keep, 0=mask) to Bool mask where True=mask
            attn_mask = attention_mask == 0  # [batch_size, seq_len]
            attn_mask = attn_mask.to(torch.bool)
        else:
            attn_mask = None

        # Added this causal mask to input
        #input_mask = Transformer.generate_square_subsequent_mask(T).to(input_ids.device)

        # Added mask and is_causal to the encoder
        output = self.encoder(x, src_key_padding_mask=attn_mask)#self.encoder(x, mask = input_mask, src_key_padding_mask=attn_mask, is_causal=True)  # [seq_len, batch_size, d_model]
        return output

class RemiDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.2,
            include_linear_head=True
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        self.include_linear_head = include_linear_head
        self.d_model = d_model
        self.vocab_size = vocab_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        if self.include_linear_head:
            self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
            self,
            input_ids,
            tgt_key_padding_mask=None,
            memory=None,
    ):
        B, T = input_ids.size()  # input: (B, T)
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)

        # causal mask
        tgt_mask = Transformer.generate_square_subsequent_mask(T).to(input_ids.device)

        if memory is None:
            memory = torch.zeros_like(x)

        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


        if self.include_linear_head:
            logits = self.out_proj(decoded)
            return logits
        else:
            return decoded

    def generate(
            self,
            bos_id,
            eos_id,
            max_len=128,
            decoding_strategy="top_p",
            top_p=0.9,
            device=None,
            memory=None,
            head_to_use=None,
    ):
        device = device or torch.device("cpu")
        generated = [bos_id]

        with torch.no_grad():
            for _ in range(1, max_len):
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                if head_to_use is not None:
                    out = self.forward(input_tensor, memory = memory)
                    logits = head_to_use(out)
                else:
                    logits = self.forward(input_tensor, memory=memory)
                last_logits = logits[0, -1]

                if decoding_strategy == "greedy":
                    next_token = int(last_logits.argmax())

                elif decoding_strategy == "top_p":
                    next_token = int(self.top_p_sample(last_logits, p=top_p))

                else:
                    raise ValueError("Unsupported decoding strategy")

                generated.append(next_token)

                if next_token == eos_id:
                    break

        return generated

    def top_p_sample(self, logits, p=0.9):
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cum_probs <= p
        mask[0] = True  # Always include at least the top token

        filtered_probs = sorted_probs[mask]
        filtered_idx = sorted_idx[mask]

        filtered_probs = filtered_probs / filtered_probs.sum()
        choice = torch.multinomial(filtered_probs, 1)
        return filtered_idx[choice].item()

class SequentialRemiDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, chord_memory, remi_memory, tgt_key_padding_mask=None):
        x = self.token_embedding(input_ids)
        x = self.pos_encoder(x)
        T = input_ids.size(1)
        tgt_mask = Transformer.generate_square_subsequent_mask(T).to(input_ids.device)
        # Concatenate chord and bass memory along sequence dimension
        memory = torch.cat([chord_memory, remi_memory], dim=1)
        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.out_proj(decoded)
        return logits

    def top_p_sample(self, logits, p=0.9):
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cum_probs <= p
        mask[0] = True  # Always include at least the top token

        filtered_probs = sorted_probs[mask]
        filtered_idx = sorted_idx[mask]

        filtered_probs = filtered_probs / filtered_probs.sum()
        choice = torch.multinomial(filtered_probs, 1)
        return filtered_idx[choice].item()

class Chord2MidiTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.memory_proj = nn.Linear(
            encoder.token_embedding.embedding_dim,
            decoder.token_embedding.embedding_dim
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        tgt,
        tgt_key_padding_mask=None,
    ):
        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)

        memory = self.memory_proj(encoder_out) # (B, T, d_dec)

        decoder_logits = self.decoder(
            tgt,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory=memory
        )

        return decoder_logits


    def generate(
        self,
        input_ids,
        attention_mask,
        bos_id,
        eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
    ):
        device = device or input_ids.device

        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)
            memory = self.memory_proj(encoder_out)

        generated_ids = self.decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
        )

        return generated_ids


class Chord2JointMidiTransformer(nn.Module):

    def __init__(self,
                 encoder,
                 bass_decoder,
                 melody_decoder,
                 #hidden_size,
                 #output_dim_a,
                 #output_dim_b
                 ):
        super().__init__()
        self.encoder = encoder
        self.bass_decoder = bass_decoder
        self.melody_decoder = melody_decoder
        self.memory_proj = nn.Linear(
            encoder.token_embedding.embedding_dim,
            bass_decoder.token_embedding.embedding_dim #same for bass and melody
        )

    def forward(self,
                input_ids,
                attention_mask,
                bass_tgt,
                melody_tgt,
                bass_tgt_key_padding_mask=None,
                melody_tgt_key_padding_mask=None,
    ):
        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)

        memory = self.memory_proj(encoder_out) # (B, T, d_dec)

        bass_decoder_out = self.bass_decoder(
            bass_tgt,
            tgt_key_padding_mask=bass_tgt_key_padding_mask,
            memory=memory,
        )

        melody_decoder_out = self.melody_decoder(
            melody_tgt,
            tgt_key_padding_mask=melody_tgt_key_padding_mask,
            memory=memory,
        )


        return bass_decoder_out, melody_decoder_out

    def generate(
        self,
        input_ids,
        attention_mask,
        bos_id,
        eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
    ):
        device = device or input_ids.device

        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)
            memory = self.memory_proj(encoder_out)

        generated_ids_bass = self.bass_decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
        )

        generated_ids_melody = self.melody_decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
        )

        return generated_ids_bass, generated_ids_melody

class Chord2JointDecoderMidiTransformer(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 vocab_size,
                 #hidden_size,
                 #output_dim_a,
                 #output_dim_b
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.memory_proj = nn.Linear(
            encoder.token_embedding.embedding_dim,
            decoder.token_embedding.embedding_dim
        )
        self.bass_head = nn.Linear(d_model, vocab_size)
        self.melody_head = nn.Linear(d_model, vocab_size)

    def forward(self,
                input_ids,
                attention_mask,
                bass_tgt,
                melody_tgt,
                bass_tgt_key_padding_mask=None,
                melody_tgt_key_padding_mask=None,
    ):
        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)

        memory = self.memory_proj(encoder_out) # (B, T, d_dec)

        bass_decoder_out = self.decoder(
            bass_tgt,
            tgt_key_padding_mask=bass_tgt_key_padding_mask,
            memory=memory,
        )

        melody_decoder_out = self.decoder(
            melody_tgt,
            tgt_key_padding_mask=melody_tgt_key_padding_mask,
            memory=memory,
        )

        bass_out = self.bass_head(bass_decoder_out)
        melody_out = self.melody_head(melody_decoder_out)

        return bass_out, melody_out

    def generate(
        self,
        input_ids,
        attention_mask,
        bos_id,
        eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
    ):
        device = device or input_ids.device

        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)
            memory = self.memory_proj(encoder_out)

        generated_ids_bass = self.decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
            head_to_use=self.bass_head,
        )

        generated_ids_melody = self.decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
            head_to_use=self.melody_head,
        )

        return generated_ids_bass, generated_ids_melody

class Chord2SequentialMidiTransformer(nn.Module):
    def __init__(
        self,
        chord_encoder,
        first_decoder,
        second_decoder,
    ):
        super().__init__()
        self.chord_encoder = chord_encoder
        self.first_decoder = first_decoder
        self.second_decoder = second_decoder

    def forward(
        self,
        chord_input_ids,
        chord_attention_mask,
        first_input_ids,
        second_input_ids,
        first_tgt_key_padding_mask=None,
        second_tgt_key_padding_mask=None
    ):
        chord_memory = self.chord_encoder(chord_input_ids, chord_attention_mask)
        first_logits = self.first_decoder(input_ids = first_input_ids, memory = chord_memory, tgt_key_padding_mask = first_tgt_key_padding_mask)
        # Use bass memory for melody decoding
        #input_ids, chord_memory, bass_memory, tgt_key_padding_mask
        first_memory = self.first_decoder.token_embedding(first_input_ids)
        second_logits = self.second_decoder(input_ids = second_input_ids, chord_memory = chord_memory, remi_memory = first_memory, tgt_key_padding_mask = second_tgt_key_padding_mask)
        return first_logits, second_logits

    def generate(
        self,
        chord_input_ids,
        chord_attention_mask,
        first_bos_id,
        first_eos_id,
        second_bos_id,
        second_eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
    ):
        device = device or chord_input_ids.device
        chord_memory = self.chord_encoder(chord_input_ids, chord_attention_mask)
        # Generate bass sequence
        first_generated = [first_bos_id]
        first_memory = None
        with torch.no_grad():
            for _ in range(1, max_len):
                first_input_tensor = torch.tensor([first_generated], dtype=torch.long, device=device)
                first_logits = self.first_decoder(input_ids = first_input_tensor, memory = chord_memory)
                last_first_logits = first_logits[0, -1]
                if decoding_strategy == "greedy":
                    next_first = int(last_first_logits.argmax())
                elif decoding_strategy == "top_p":
                    next_first = int(self.first_decoder.top_p_sample(last_first_logits, p=top_p))
                else:
                    raise ValueError("Unsupported decoding strategy")
                first_generated.append(next_first)
                if next_first == first_eos_id:
                    break
            first_memory = self.first_decoder.token_embedding(torch.tensor([first_generated], dtype=torch.long, device=device))
        # Generate melody sequence conditioned on chord and bass
        second_generated = [second_bos_id]
        with torch.no_grad():
            for _ in range(1, max_len):
                second_input_tensor = torch.tensor([second_generated], dtype=torch.long, device=device)
                second_logits = self.second_decoder(input_ids = second_input_tensor, chord_memory = chord_memory, remi_memory = first_memory)
                last_second_logits = second_logits[0, -1]
                if decoding_strategy == "greedy":
                    next_second = int(last_second_logits.argmax())
                elif decoding_strategy == "top_p":
                    next_second = int(self.second_decoder.top_p_sample(last_second_logits, p=top_p))
                else:
                    raise ValueError("Unsupported decoding strategy")
                second_generated.append(next_second)
                if next_second == second_eos_id:
                    break
        return first_generated, second_generated
