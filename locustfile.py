from locust import HttpUser, task, between

class VirusClassificationLoadTest(HttpUser):
    # time each simulated user waits between requests
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        """Send a prediction request with a sample image."""
        # Ensure you have a sample.jpg in the same directory when running locust
        with open("sample.jpg", "rb") as img:
            self.client.post(
                "/predict",
                files={"file": ("sample.jpg", img, "image/jpeg")}
            )