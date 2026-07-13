import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple


def normalize_relation(relation: str, task: str) -> str:
    relation = str(relation).strip()
    if task == "WN18RR":
        return relation.replace("_", " ").strip()
    if task == "FB15k237":
        tokens = relation.replace("./", "/").replace("_", " ").strip("/").split("/")
        return " ".join(token.strip() for token in tokens if token.strip())
    return relation.replace("_", " ").strip()


def read_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Invalid triple line {line_no} in {path}: {line}")
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def load_wn_text(data_dir: str) -> Dict[str, Tuple[str, str]]:
    path = os.path.join(data_dir, "wordnet-mlj12-definitions.txt")
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                entity_id = fields[0]
                name = fields[1].replace("__", " ").replace("_", " ").strip()
                desc = fields[2].strip()
                result[entity_id] = (name or entity_id, desc)
    return result


def _load_two_column(path: str) -> Dict[str, str]:
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            fields = line.rstrip("\n").split("\t", 1)
            if len(fields) == 2:
                result[fields[0]] = fields[1]
    return result


def load_fb_text(data_dir: str) -> Dict[str, Tuple[str, str]]:
    names = _load_two_column(os.path.join(data_dir, "FB15k_mid2name.txt"))
    descs = _load_two_column(os.path.join(data_dir, "FB15k_mid2description.txt"))
    result = {}
    for entity_id in set(names) | set(descs):
        name = names.get(entity_id, entity_id).replace("_", " ").strip()
        desc = " ".join(descs.get(entity_id, "").split()[:50])
        result[entity_id] = (name or entity_id, desc)
    return result


def load_custom_text(data_dir: str) -> Dict[str, Tuple[str, str]]:
    path = os.path.join(data_dir, "entity_text.tsv")
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                result[fields[0]] = (fields[1], fields[2] if len(fields) >= 3 else "")
    return result


def main():
    parser = argparse.ArgumentParser("Preprocess WN18RR/FB15k237 for SIEVE-KGC")
    parser.add_argument("--task", default="WN18RR", choices=["WN18RR", "FB15k237", "custom"])
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    if args.task == "WN18RR":
        text_map = load_wn_text(args.data_dir)
    elif args.task == "FB15k237":
        text_map = load_fb_text(args.data_dir)
    else:
        text_map = load_custom_text(args.data_dir)

    split_triples = {}
    entity_ids = OrderedDict()
    relation_map = OrderedDict()
    for split in ("train", "valid", "test"):
        raw_path = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)
        triples = read_triples(raw_path)
        split_triples[split] = triples
        for h, r, t in triples:
            entity_ids.setdefault(h, None)
            entity_ids.setdefault(t, None)
            relation_map.setdefault(r, normalize_relation(r, args.task))

    entities = []
    for entity_id in entity_ids:
        name, desc = text_map.get(entity_id, (entity_id.replace("_", " "), ""))
        entities.append({
            "entity_id": entity_id,
            "entity": name,
            "entity_desc": desc,
        })
    with open(os.path.join(args.data_dir, "entities.json"), "w", encoding="utf-8") as writer:
        json.dump(entities, writer, ensure_ascii=False, indent=2)
    with open(os.path.join(args.data_dir, "relations.json"), "w", encoding="utf-8") as writer:
        json.dump(relation_map, writer, ensure_ascii=False, indent=2)

    entity_text = {obj["entity_id"]: (obj["entity"], obj["entity_desc"]) for obj in entities}
    for split, triples in split_triples.items():
        examples = []
        for h, r, t in triples:
            h_name, h_desc = entity_text[h]
            t_name, t_desc = entity_text[t]
            examples.append({
                "head_id": h,
                "head": h_name,
                "head_desc": h_desc,
                "relation": relation_map[r],
                "tail_id": t,
                "tail": t_name,
                "tail_desc": t_desc,
            })
        out_path = os.path.join(args.data_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(examples, writer, ensure_ascii=False, indent=2)
        print(f"Saved {len(examples)} examples to {out_path}")
    print(f"Saved {len(entities)} entities to {args.data_dir}/entities.json")


if __name__ == "__main__":
    main()
