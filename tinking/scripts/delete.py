"""Delete every ckpt on your Tinker account"""

import tinker
from tqdm import tqdm

def main():
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    # Get all checkpoints in one go (suitable for moderate numbers)
    checkpoints = []
    offset = 0
    limit = 100
    while True:
        resp = rest_client.list_user_checkpoints(limit=limit, offset=offset).result()
        if not resp.checkpoints:
            break
        checkpoints.extend(resp.checkpoints)
        if resp.cursor and offset + limit < resp.cursor.total_count:
            offset += limit
        else:
            break

    print(f"Found {len(checkpoints)} checkpoints...")

    for cp in tqdm(checkpoints, desc="Deleting checkpoints"):
        rest_client.delete_checkpoint_from_tinker_path(cp.tinker_path).result()

if __name__ == "__main__":
    main()
