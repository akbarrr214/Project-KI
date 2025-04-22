import streamlit as st
import numpy as np  # Pastikan NumPy diimpor terlebih dahulu
import io
import base64
from PIL import Image

# Import cv2 setelah NumPy
import cv2
from utils.image_utils import embed_message_dct, extract_message_dct, calculate_psnr, calculate_mse, calculate_ssim

# Set page configuration
st.set_page_config(
    page_title="Silentium",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist design
def apply_custom_css():
    st.markdown("""
    <style>
        /* Base styling */
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
            color: #1A1A1A;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #1A1A1A;
            font-weight: 600;
            letter-spacing: -0.03em;
        }
        
        /* Header */
        .sv-header {
            padding: 2rem 0;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .sv-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1A1A1A;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        
        .sv-subtitle {
            font-size: 1.1rem;
            color: #666666;
            margin-bottom: 2rem;
        }
        
        /* Navigation */
        .sv-nav {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 3rem;
            padding: 0.5rem;
            background: #F5F5F5;
            border-radius: 12px;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
        
        .sv-nav-item {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.95rem;
            color: #666666;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .sv-nav-item:hover {
            background: #EEEEEE;
        }
        
        .sv-nav-item.active {
            background: #1A1A1A;
            color: #FFFFFF;
        }
        
        /* Cards */
        .sv-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #EEEEEE;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        /* Section headers */
        .sv-section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #1A1A1A;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #333333 !important;
            transform: translateY(-1px) !important;
        }
        
        /* Form elements */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: #F5F5F5 !important;
            border: 1px solid #EEEEEE !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            color: #1A1A1A !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #1A1A1A !important;
            box-shadow: 0 0 0 1px #1A1A1A !important;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #F5F5F5 !important;
            border: 2px dashed #DDDDDD !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
        }
        
        /* Metrics */
        .sv-metrics {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .sv-metrics-card {
            flex: 1;
            background: #F5F5F5;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .sv-metrics-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1A1A1A;
            margin: 0.5rem 0;
        }
        
        .sv-metrics-label {
            font-size: 0.9rem;
            color: #666666;
        }
        
        /* Alert boxes */
        .sv-alert {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }
        
        .sv-alert-info {
            background-color: #EEF6FF;
            border: 1px solid #CCE4FF;
            color: #0066CC;
        }
        
        .sv-alert-success {
            background-color: #EEFFF3;
            border: 1px solid #CCFFD8;
            color: #00802B;
        }
        
        .sv-alert-error {
            background-color: #FFF2F0;
            border: 1px solid #FFD6D0;
            color: #CC1100;
        }
        
        .sv-alert-warning {
            background-color: #FFF8E6;
            border: 1px solid #FFE8B3;
            color: #996600;
        }
        
        /* Message display */
        .sv-message {
            background-color: #F5F5F5;
            border-radius: 8px;
            padding: 1.5rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
            font-size: 0.9rem;
            margin: 1.5rem 0;
            overflow-x: auto;
        }
        
        /* Progress */
        .sv-progress {
            height: 6px;
            background: #EEEEEE;
            border-radius: 3px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .sv-progress-bar {
            height: 100%;
            background: #1A1A1A;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        /* Footer */
        .sv-footer {
            text-align: center;
            margin-top: 4rem;
            padding: 2rem 0;
            font-size: 0.9rem;
            color: #666666;
            border-top: 1px solid #EEEEEE;
        }
        
        /* Hide Streamlit default elements */
        .stDeployButton, footer {
            display: none !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .sv-title {
                font-size: 2rem;
            }
            
            .sv-metrics {
                flex-direction: column;
            }
            
            .sv-card {
                padding: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Session state initialization
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'embed'

# Header
st.markdown("""
<div class="sv-header">
    <div class="sv-title">
        <span>🔒</span>
        <span>Silentium</span>
    </div>
    <div class="sv-subtitle">A cutting-edge steganography platform that hides messages in visual silence.</div>
</div>
""", unsafe_allow_html=True)

# Navigation
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col2:
    st.markdown("""
    <div class="sv-nav">
        <div class="sv-nav-item {active_embed}" onclick="document.getElementById('btn-embed').click();">Embed Data</div>
        <div class="sv-nav-item {active_extract}" onclick="document.getElementById('btn-extract').click();">Extract Data</div>
        <div class="sv-nav-item {active_about}" onclick="document.getElementById('btn-about').click();">About</div>
    </div>
    """.format(
        active_embed='active' if st.session_state['current_page'] == 'embed' else '',
        active_extract='active' if st.session_state['current_page'] == 'extract' else '',
        active_about='active' if st.session_state['current_page'] == 'about' else ''
    ), unsafe_allow_html=True)

# Hidden navigation buttons
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Embed", key="btn-embed", on_click=lambda: st.session_state.update({'current_page': 'embed'}), 
              use_container_width=True)
with col2:
    st.button("Extract", key="btn-extract", on_click=lambda: st.session_state.update({'current_page': 'extract'}),
              use_container_width=True)
with col3:
    st.button("About", key="btn-about", on_click=lambda: st.session_state.update({'current_page': 'about'}),
              use_container_width=True)

# Embed Page
if st.session_state['current_page'] == 'embed':
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-section-title">Hide Secret Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sv-alert sv-alert-info">
        <strong>How it works:</strong> Your message will be encoded into the image using DCT. 
        The changes are imperceptible to the human eye.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Choose carrier image", type=["jpg", "jpeg", "png"], key="embed_uploader")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            img_width, img_height = image.size
            max_chars = (img_width * img_height) // 64 // 8
            
            st.markdown(f"""
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #666666;">
                Maximum capacity: {max_chars} characters
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        message = st.text_area("Secret message", placeholder="Type your secret message here...", height=150, key="embed_message")
        
        if message and uploaded_file:
            char_count = len(message)
            percentage = min(100, char_count / max_chars * 100)
            
            st.markdown(f"""
            <div style="text-align: right; font-size: 0.9rem; color: #666666; margin-bottom: 0.5rem;">
                {char_count} / {max_chars} characters ({percentage:.1f}%)
            </div>
            <div class="sv-progress">
                <div class="sv-progress-bar" style="width: {percentage}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        embed_button = st.button("Hide Secret Data", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file and message and embed_button:
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-section-title">Results</div>', unsafe_allow_html=True)
        
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        with st.spinner("Processing..."):
            stego_img, success = embed_message_dct(img_array, message)
        
        if success:
            stego_pil = Image.fromarray(stego_img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="text-align: center; margin-bottom: 1rem;">Original Image</div>', unsafe_allow_html=True)
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown('<div style="text-align: center; margin-bottom: 1rem;">Steganographic Image</div>', unsafe_allow_html=True)
                st.image(stego_pil, use_column_width=True)
            
            # Metrics
            psnr = calculate_psnr(img_array, stego_img)
            mse = calculate_mse(img_array, stego_img)
            ssim_value = calculate_ssim(img_array, stego_img)
            
            st.markdown('<div class="sv-metrics">', unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.markdown(f"""
                <div class="sv-metrics-card">
                    <div class="sv-metrics-label">PSNR</div>
                    <div class="sv-metrics-value">{psnr:.2f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="sv-metrics-card">
                    <div class="sv-metrics-label">MSE</div>
                    <div class="sv-metrics-value">{mse:.6f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="sv-metrics-card">
                    <div class="sv-metrics-label">SSIM</div>
                    <div class="sv-metrics-value">{ssim_value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sv-alert sv-alert-success">
                <strong>Success!</strong> Your message has been successfully hidden in the image.
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            buf = io.BytesIO()
            stego_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Steganographic Image",
                data=byte_im,
                file_name="silentium_image.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="sv-alert sv-alert-error">
                <strong>Error:</strong> The message is too large for this image. Please use a larger image or reduce your message.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Extract Page
elif st.session_state['current_page'] == 'extract':
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-section-title">Extract Hidden Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sv-alert sv-alert-warning">
        <strong>Important:</strong> Use the original steganographic image without modifications for reliable extraction.
    </div>
    """, unsafe_allow_html=True)
    
    stego_file = st.file_uploader("Upload steganographic image", type=["jpg", "jpeg", "png", "gif"], key="extract_uploader")
    
    if stego_file is not None:
        stego_image = Image.open(stego_file)
        st.image(stego_image, use_column_width=True)
    
    extract_button = st.button("Extract Hidden Data", use_container_width=True, disabled=stego_file is None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if stego_file and extract_button:
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        
        stego_array = np.array(stego_image)
        if len(stego_array.shape) == 3 and stego_array.shape[2] == 4:
            stego_array = cv2.cvtColor(stego_array, cv2.COLOR_RGBA2RGB)
        
        with st.spinner("Extracting..."):
            extracted_message = extract_message_dct(stego_array)
        
        if extracted_message:
            st.markdown("""
            <div class="sv-alert sv-alert-success">
                <strong>Success!</strong> A hidden message was found.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="sv-message">', unsafe_allow_html=True)
            st.markdown(f"{extracted_message}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="Download Message",
                data=extracted_message,
                file_name="extracted_message.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="sv-alert sv-alert-error">
                <strong>No message found:</strong> This image may not contain hidden data or it may have been corrupted.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# About Page
elif st.session_state['current_page'] == 'about':
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-section-title">About DCT Steganography</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Silentium uses Discrete Cosine Transform (DCT) based steganography to hide messages within images. 
    This technique is robust and imperceptible to the human eye.
    
    ### How it works:
    1. The image is divided into 8×8 pixel blocks
    2. Each block is transformed using DCT
    3. Message bits are embedded in specific frequency coefficients
    4. Inverse DCT reconstructs the modified image
    
    ### Key advantages:
    - *Imperceptible:* Changes are virtually invisible
    - *Robust:* Resistant to compression and minor modifications
    - *Efficient:* Can store significant data relative to image size
    
    ### Best practices:
    - Use complex images with varied textures
    - Keep messages concise for better security
    - Avoid image manipulation after embedding
    - Always test extraction before relying on stego images
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="sv-footer">
    Silentium © 2025 | Created by Information Security Group
</div>
""", unsafe_allow_html=True)