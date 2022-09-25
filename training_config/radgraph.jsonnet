local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-uncased",
  cuda_device: -1,
  data_paths: {
    train: "data/train.json",
    validation: "data/dev.json",
    test: "data/test.json"
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
}