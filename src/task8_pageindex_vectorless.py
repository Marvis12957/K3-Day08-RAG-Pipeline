"""Task 8 - PageIndex fallback hook.

The class checkpoint allows PageIndex to be unavailable. Returning an empty
list keeps Task 9 from falling back to noisy hybrid results for off-domain
queries when no PageIndex index/key is configured.
"""


def upload_documents():
    raise NotImplementedError("PageIndex upload requires a PAGEINDEX_API_KEY")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    return []


if __name__ == "__main__":
    print("PageIndex is not configured; fallback returns [].")
