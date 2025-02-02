import streamlit as st
import torch
from my_model import Seq2SeqTransformer, Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadAttentionLayer, PositionwiseFeedforwardLayer
import torch.nn.functional as F
import numpy as np
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Load the vocab from the saved file
vocab_transform = torch.load('vocab')

# Access the vocab objects for each language
SRC_VOCAB = vocab_transform['en']
TRG_VOCAB = vocab_transform['ur']

# Access the special token indices directly
SRC_PAD_IDX = SRC_VOCAB['<pad>']
TRG_PAD_IDX = TRG_VOCAB['<pad>']
SOS_IDX = SRC_VOCAB['<sos>']
EOS_IDX = SRC_VOCAB['<eos>']

# Hyperparameters
SRC_VOCAB_SIZE = len(SRC_VOCAB)
TRG_VOCAB_SIZE = len(TRG_VOCAB)
DEVICE = torch.device('mps')

# Load the trained model
attn_variant = "general"
enc = Encoder(SRC_VOCAB_SIZE, 256, 3, 8, 512, 0.1, attn_variant, DEVICE)
dec = Decoder(TRG_VOCAB_SIZE, 256, 3, 8, 512, 0.1, attn_variant, DEVICE)

# Initialize model
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)

# Load the saved model checkpoint (the checkpoint is a list)
checkpoint = torch.load('general_transformer.pt', map_location=DEVICE)

# Check if the checkpoint is a list and handle it accordingly
if isinstance(checkpoint, list):
    state_dict = checkpoint[0]  # Assuming the first element in the list is the state_dict
else:
    state_dict = checkpoint  # If it's already a dictionary, use it directly

# Load the state_dict into the model
model.load_state_dict(state_dict, strict=False)

# Function to translate input sentence
def translate_sentence(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    src_vocab = vocab_transform['en']
    trg_vocab = vocab_transform['sd']
    src_stoi = src_vocab.get_stoi()
    trg_stoi = trg_vocab.get_stoi()
    trg_itos = trg_vocab.get_itos()

    src = [src_stoi[word] if word in src_stoi else src_stoi['<unk>'] for word in sentence.split()]
    src = torch.tensor(src).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    src_mask = model.make_src_mask(src)

    # Count input sentence length and determine the desired length for the output
    input_len = len(sentence.split())
    output_len = max(1, input_len + np.random.randint(-3, 4))  # Allow output to be +/- 3 words from input length

    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
        trg = torch.tensor([trg_stoi['<sos>']]).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg)

        for _ in range(output_len):  # Limit the output length to the calculated value
            output, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()

            if pred_token == trg_stoi['<eos>']:
                break

            trg = torch.cat((trg, torch.tensor([[pred_token]]).to(device)), dim=1)
            trg_mask = model.make_trg_mask(trg)

    trg_tokens = trg.squeeze(0).cpu().numpy()
    trg_sentence = ' '.join([trg_itos[i] for i in trg_tokens])
    return trg_sentence

# Streamlit app
st.set_page_config(page_title="Translation App", page_icon="🌐")  # Add an icon to the app
st.markdown("""
    <style>
    .header {
        color: #4CAF50;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        padding-bottom: 20px;
    }
    .output {
        background-color: #f4f4f4;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.2em;
        color: #333;
        margin-top: 20px;
    }
    .input-container {
        background-color: #f0f0f5;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.1em;
        color: #555;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header">📜 English to Urdu Translation </div>', unsafe_allow_html=True)

sentence = st.text_area("Enter your sentence in English:", height=204, max_chars=300) 

# Translate button with a spinner for processing
if st.button("Translate"):
    if sentence:
        with st.spinner("Translating... Please wait! 🌀"):
            translated_sentence = translate_sentence(sentence, model, vocab_transform, DEVICE)
        
        # Display translation result with a custom container and style
        st.markdown('<div class="output"><b>Translation:</b><br><i>{}</i></div>'.format(translated_sentence), unsafe_allow_html=True)
    else:
        st.error("Please enter a sentence to translate.")
