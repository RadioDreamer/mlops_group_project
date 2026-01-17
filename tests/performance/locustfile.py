from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


class APIUser(HttpUser):
    """
    Performance test for the FakeArtDetector API.
    Optimized to minimize generator-side overhead.
    """

    wait_time = between(1, 4)

    def on_start(self):
        """
        Runs once per virtual user.
        Pre-generates the image bytes to isolate API performance.
        """
        img = Image.new("RGB", (32, 32), color="green")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        self.image_bytes = buf.getvalue()

    @task(1)
    def health_check(self):
        self.client.get("/")

    @task(3)
    def predict_image(self):
        """Sends pre-generated image bytes to the inference endpoint."""
        files = {"data": ("load_test.jpg", self.image_bytes, "image/jpeg")}
        self.client.post("/model/", files=files)
