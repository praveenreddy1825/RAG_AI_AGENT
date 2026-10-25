"""
Utility Script: Sync Local Qdrant Vectors to Qdrant Cloud.

Copies all vectors, payloads, and public documents from local disk storage
(./qdrant_storage) directly to your remote Qdrant Cloud cluster without re-embedding.

Usage:
  uv run python sync_to_cloud.py --url https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io:6333 --api-key <API_KEY>
"""

import sys
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import settings

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sync_vectors(cloud_url: str, cloud_api_key: str, collection_name: str = None):
    collection = collection_name or settings.QDRANT_COLLECTION_NAME
    local_path = str(Path(settings.QDRANT_STORAGE_PATH).resolve())

    print(f"\n🚀 Qdrant Vector Sync to Cloud")
    print(f"==================================================")
    print(f"📦 Source (Local):   {local_path}")
    print(f"☁️ Destination:      {cloud_url}")
    print(f"📚 Collection:       {collection}")
    print(f"==================================================\n")

    # 1. Connect to local Qdrant
    print("⏳ Opening local Qdrant storage...")
    local_client = QdrantClient(path=local_path)
    if not local_client.collection_exists(collection):
        print(f"❌ Error: Local collection '{collection}' not found in {local_path}!")
        return

    local_info = local_client.get_collection(collection)
    total_points = local_info.points_count
    print(f"✅ Local collection found! Total vectors to transfer: {total_points}")

    # 2. Connect to Cloud Qdrant
    print(f"\n⏳ Connecting to Qdrant Cloud...")
    try:
        cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key, timeout=30)
        cloud_client.get_collections()
        print("✅ Successfully connected to Qdrant Cloud cluster!")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud: {e}")
        return

    # 3. Ensure collection exists in Cloud
    if not cloud_client.collection_exists(collection):
        print(f"⚙️ Creating collection '{collection}' (dim={settings.EMBEDDING_DIM}, Cosine) in Cloud...")
        cloud_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print("✅ Cloud collection created successfully.")
    else:
        print(f"ℹ️ Cloud collection '{collection}' already exists.")

    # Ensure payload keyword indices exist in Cloud
    for field in ["user_id", "source_id"]:
        try:
            cloud_client.create_payload_index(collection, field, "keyword")
        except Exception:
            pass

    # 4. Scroll local points and batch upload to Cloud
    print("\n⏳ Transferring vectors and metadata payloads...")
    batch_size = 100
    offset = None
    transferred = 0

    while True:
        records, next_offset = local_client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        points_to_upsert = [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload=record.payload,
            )
            for record in records
        ]

        cloud_client.upsert(
            collection_name=collection,
            points=points_to_upsert,
        )

        transferred += len(records)
        print(f"  ➡️ Transferred {transferred}/{total_points} vectors ({(transferred/total_points)*100:.1f}%)")

        if next_offset is None:
            break
        offset = next_offset

    cloud_info = cloud_client.get_collection(collection)
    print(f"\n🎉 Sync Complete! Cloud collection now contains {cloud_info.points_count} vectors.")
    print("Your Streamlit Cloud deployment can now query these public documents immediately!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync local Qdrant vectors to Qdrant Cloud.")
    parser.add_argument("--url", help="Qdrant Cloud Cluster URL (e.g. https://xxx.cloud.qdrant.io:6333)")
    parser.add_argument("--api-key", help="Qdrant Cloud API Key")
    parser.add_argument("--collection", default=None, help="Collection name (default: rag_documents)")
    args = parser.parse_args()

    url = args.url or settings.QDRANT_URL
    api_key = args.api_key or settings.QDRANT_API_KEY

    if not url or "localhost" in url:
        url = input("Enter your Qdrant Cloud Cluster URL: ").strip()
    if not api_key:
        api_key = input("Enter your Qdrant Cloud API Key: ").strip()

    if not url or not api_key:
        print("❌ URL and API Key are required to sync to Qdrant Cloud.")
        sys.exit(1)

    sync_vectors(url, api_key, args.collection)
