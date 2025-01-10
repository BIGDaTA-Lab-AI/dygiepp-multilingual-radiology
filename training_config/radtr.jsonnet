local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "pretrained/BioBERTurk",
  data_paths: {
    train: "data/radtr/train.json",
    validation: "data/radtr/dev.json",
    test: "data/radtr/test.json"
  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "ner",
  max_span_width: 4,
  trainer: {
      checkpointer: {
        
      },
      num_epochs: 150,
      grad_norm: 5.0,
      cuda_device: -1,
      validation_metric: '+MEAN__ner_f1',
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