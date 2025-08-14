import requests
from typing import List

class BrightDataTrigger:
    def __init__(self, api_token: str, dataset_id: str):
        self.api_token = api_token
        self.dataset_id = dataset_id
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    def create_snapshot_from_usernames(self, usernames: List[str]) -> str:
        payload = [{"user_name": u} for u in usernames]
        params = {
            "dataset_id": self.dataset_id,
            "type": "discover_new",
            "discover_by": "user_name"
        }
        resp = requests.post(f"{self.base_url}/trigger", headers=self.headers, params=params, json=payload)
        resp.raise_for_status()
        return resp.json().get("snapshot_id")
