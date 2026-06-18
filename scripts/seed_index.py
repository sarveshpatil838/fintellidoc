"""
Seed the FAISS index with sample financial documents.
Run: python scripts/seed_index.py

This script indexes a few sample documents so you can demo the /query endpoint
without needing to upload documents first.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag import RAGService

SAMPLE_DOCS = [
    {
        "doc_id": "aapl-q3-2024",
        "text": """Apple Inc. Q3 2024 Earnings Release

Apple Inc. (AAPL) today announced financial results for its fiscal 2024 third quarter 
ended June 29, 2024. The Company posted quarterly revenue of $85.8 billion, up 5 percent 
year over year, and quarterly earnings per diluted share of $1.40, up 11 percent year over year.

"We are happy to report that we had an all-time revenue record in Services and a June quarter 
record for iPhone," said Tim Cook, Apple's CEO. "We are also thrilled to be bringing Apple 
Intelligence to users very soon, with a personal intelligence system that puts powerful, 
private AI features at the fingertips of our users."

Revenue by segment:
- iPhone: $39.3 billion, up 3% YoY
- Mac: $7.0 billion, up 2% YoY  
- iPad: $7.2 billion, up 24% YoY
- Wearables, Home and Accessories: $8.1 billion, down 2% YoY
- Services: $24.2 billion, up 14% YoY (all-time record)

Gross margin: 46.3%
Operating expenses: $14.5 billion
Net income: $21.4 billion
""",
        "metadata": {"company": "Apple Inc.", "period": "Q3 2024", "type": "earnings"}
    },
    {
        "doc_id": "msft-q4-2024",
        "text": """Microsoft Corporation Q4 FY2024 Earnings

Microsoft Corporation (MSFT) reported the following results for the quarter ended June 30, 2024:

Revenue was $64.7 billion and increased 15% (up 15% in constant currency).
Operating income was $27.9 billion and increased 23% (up 24% in constant currency).
Net income was $22.0 billion and increased 10% (up 10% in constant currency).
Diluted earnings per share was $2.95 and increased 10% (up 10% in constant currency).

Cloud revenue was $36.8 billion, up 21% year-over-year.
Microsoft Cloud gross margin percentage increased to 72%.

Satya Nadella, chairman and chief executive officer of Microsoft, stated: "We are 
accelerating innovation and growth in Azure with our differentiated AI platform."

Azure and other cloud services revenue grew 29% year-over-year.
""",
        "metadata": {"company": "Microsoft Corp.", "period": "Q4 FY2024", "type": "earnings"}
    },
]


async def main():
    print("Initializing RAG service...")
    service = RAGService()

    for doc in SAMPLE_DOCS:
        print(f"Indexing {doc['doc_id']}...")
        chunks = await service.index_document(doc["text"], doc["doc_id"], doc["metadata"])
        print(f"  → {chunks} chunks indexed")

    print("\nIndex seeded successfully!")
    print(f"Total chunks in index: {service.index.ntotal}")
    print("\nTry querying:")
    print('  POST /api/v1/query {"question": "What was Apple revenue in Q3 2024?"}')
    print('  POST /api/v1/query {"question": "How did Microsoft Azure grow?"}')


if __name__ == "__main__":
    asyncio.run(main())
