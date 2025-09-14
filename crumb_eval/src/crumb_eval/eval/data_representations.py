from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Item:
    content: str | None = None
    token_ids: List[int] | None = None
    id: str | None = None
    metadata: Dict | None = None

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        metadata_key: str | None = None,
        content_key: str | None = None,
        id_key: str | None = None,
    ) -> "Item":
        metadata_key = metadata_key if metadata_key is not None else "metadata"
        content_key = content_key if content_key is not None else "content"
        id_key = id_key if id_key is not None else "id"

        return cls(
            content=data.get(content_key),
            token_ids=data.get("token_ids"),
            id=data.get(id_key),
            metadata=data.get(metadata_key),
        )

    def to_dict(
        self,
        metadata_key: str | None = None,
        content_key: str | None = None,
        id_key: str | None = None,
    ) -> Dict:
        metadata_key = metadata_key if metadata_key is not None else "metadata"
        content_key = content_key if content_key is not None else "content"
        id_key = id_key if id_key is not None else "id"

        data = {}

        if self.content is not None:
            data[content_key] = self.content

        if self.token_ids is not None:
            data["token_ids"] = self.token_ids

        if self.id is not None:
            data[id_key] = self.id

        if self.metadata is not None:
            data[metadata_key] = self.metadata

        return data


@dataclass
class QueryAssociatedItems:
    query: Item
    items: List[Item]
    item_scores: List[float] | None = None
    metadata: Dict | None = None

    @classmethod
    def from_dict(
        cls, data: Dict, score_key: str = "score"
    ) -> "QueryAssociatedItems":
        return cls(
            query=Item.from_dict(data["query"]),
            items=[Item.from_dict(item) for item in data["items"]],
            item_scores=[item.get(score_key) for item in data["items"]],
            metadata=data.get("metadata", None),
        )

    def to_dict(self, score_key: str = "score") -> Dict:
        if self.item_scores is None:
            items = [item.to_dict() for item in self.items]
        else:
            items = [
                {**item.to_dict(), score_key: score}
                for item, score in zip(self.items, self.item_scores)
            ]

        output = {
            "query": self.query.to_dict(),
            "items": items,
        }

        if self.metadata is not None:
            output["metadata"] = self.metadata

        return output
