from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


class APIUser(HttpUser):
    # Time to wait between tasks (simulates real user behavior)
    wait_time = between(1, 5)

    @task(1)
    def health_check(self):
        """Standard task to check API health."""
        self.client.get("/")

    @task(3)
    def predict_image(self):
        """More frequent task simulating image inference."""
        img = Image.new("RGB", (32, 32), color="green")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")

        files = {"data": ("load_test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
        self.client.post("/model/", files=files)
