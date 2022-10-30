local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "models/PubMedBERT-base-uncased",
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
  trainer: {
      checkpointer: {
        num_serialized_models_to_keep: 3,
      },
      num_epochs: 50,
      grad_norm: 5.0,
      cuda_device: -1,
      validation_metric: '+MEAN__relation_f1',
      optimizer: {
        type: 'adamw',
        lr: 1e-3,
        weight_decay: 0.0,
        parameter_groups: [
          [
            ['_embedder'],
            {
              lr: 5e-5,
              weight_decay: 0.01,
              finetune: true,
            },
          ],
        ],
      },
      learning_rate_scheduler: {
        type: 'slanted_triangular'
      }
    },
}