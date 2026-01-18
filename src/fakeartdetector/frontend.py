import io
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from google.cloud import run_v2

# Load environment variables from .env file
load_dotenv()


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    # For local development, uncomment the line below
    # return 'http://127.0.0.1:8000'

    # Check environment variable first
    env_backend = os.environ.get("BACKEND")
    if env_backend:
        print(f"Using BACKEND from .env: {env_backend}")
        return env_backend

    # # Try to discover Cloud Run service automatically
    try:
        parent = "projects/double-zenith-484209-d9/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        print("Discovered Cloud Run services:")
        for service in services:
            service_name = service.name.split("/")[-1]
            print(f"  - {service_name}: {service.uri}")
            if service_name == "fake-art-api":
                discovered_url = str(service.uri)  # Convert to string
                print(f"Using discovered service: {discovered_url}")
                return discovered_url
    except Exception as e:
        print(f"Could not discover Cloud Run service: {e}")
    return None


def classify_image(image: bytes, backend: str, filename: str | None = None, mime: str | None = None):
    """Send the image to the backend for classification.

    The API expects a multipart-form field named "data" (UploadFile).
    """
    predict_url = f"{backend}/model/"
    file_tuple = (filename or "image", io.BytesIO(image), mime or "application/octet-stream")
    response = requests.post(predict_url, files={"data": file_tuple}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(
            image,
            backend=backend,
            filename=getattr(uploaded_file, "name", None),
            mime=getattr(uploaded_file, "type", None),
        )

        if result is not None:
            is_ai = result.get("isAI")
            prob_ai = float(result.get("probability", 0.0))

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", "AI" if is_ai else "Human")

            # two-class probability bar chart (AI vs Human)
            data = {"Class": ["AI", "Human"], "Probability": [prob_ai, max(0.0, 1.0 - prob_ai)]}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
