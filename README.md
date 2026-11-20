# NeuroFuse-AD

A research prototype for federated learning with retinal images, speech transcripts and time-ordered EHR events. The runnable demonstration learns a binary **synthetic** label; it has no clinical reference cohort or validated Alzheimer’s prediction endpoint.

## Run the explorer

From the extracted project folder:

    python -m http.server 8000 --directory docs

Open http://localhost:8000. Select one of eight held-out example records and toggle any input. The explorer displays model outputs exported by the training run for all seven nonempty combinations. Clearing all inputs abstains. The site serves static assets and accepts no patient uploads.

## Reproduce the synthetic experiment

Use Python 3.11 or 3.12 in a virtual environment.

    python -m pip install -r requirements.txt
    python -m unittest -v
    python train.py

The small network trains from scratch: a patch-based ViT, a bidirectional BERT-style transcript encoder, a time-decayed graph over consecutive visits, masked attention and a binary head. This model uses a fixed eight-word vocabulary and generated image patterns. Each full-batch FedAvg round trains independent client models on 48, 60 and 72 synthetic records, then averages weights by client size. The fourth split contains 96 independently generated records. All clients use the same generator family. The demo estimates performance on this toy task only.

The default experiment uses seed 23, 12 rounds, three local steps, SGD at 0.12 and CPU execution. It saves weights to docs/experiment.pt and scores to docs/experiment.json. Replay on the held-out cohort with python:

    from train import cohort, evaluate
    from model import NeuroFuse
    import torch
    model = NeuroFuse()
    model.load_state_dict(torch.load("docs/experiment.pt", weights_only=True))
    cases, metrics = evaluate(model.eval(), cohort(96, 3, 95))

Optional synthetic clipping/noise experiment:

    python train.py --noise 1 --local-steps 1 --rounds 1 --output private_data/noisy.json

Each local step clips each record’s full gradient to L2 norm 1, perturbs the sum with Gaussian noise and divides by the local sample count. The noise generator uses a fresh seed. This experimental path has no audited privacy accountant, secure aggregation or formal privacy claim. The separately runnable site client below uses conventional SGD.

## Research pipeline with pretrained encoders

    python -m pip install -r requirements-site.txt

site_training.py provides a frozen Hugging Face ViT for RGB images and a frozen BERT model for text, paired with trainable projections, a temporal graph encoder and a fusion head. It accepts variable-length transcripts and visit histories. Model downloads require internet on first use; pass local directories with --vision and --language before the subcommand to run offline. Pin the same weights, preprocessing and tokenizer at all sites. The demo weights and the pretrained research model have separate architectures and checkpoints.

Each site keeps one JSON object per line in a local manifest. Example schema (invented values):

    {"id":"local-001","image":"images/example.png","transcript":"the window opens","visits":[[0,0.2,-0.1],[2,0.3,0.0]],"label":1}

Each visit contains elapsed time and two standardized study-defined features. Omit any absent branch. Align inputs by participant and index time and use a patient-disjoint test manifest. Define the clinical reference label, feature definitions, time unit and preprocessing from the training cohort before fitting a research model.

    python site_training.py init --out initial.safetensors
    python site_training.py client --site A --manifest private_data/a.jsonl --base initial.safetensors --out a.safetensors
    python site_training.py client --site B --manifest private_data/b.jsonl --base initial.safetensors --out b.safetensors
    python site_training.py aggregate --base initial.safetensors --updates a.safetensors b.safetensors --out round1.safetensors
    python site_training.py evaluate --base round1.safetensors --manifest private_data/test.jsonl

Run each client command inside its institution. Move only its safetensors file and matching JSON metadata to the coordinator through an authorized channel. Distribute the aggregated weights as the base for the next round. The coordinator requires no images, transcripts or EHRs. Update files can still expose information; authenticated transport, secure aggregation, clinical validation and governance are outside this prototype.

## GitHub Pages

Publish the source to the main branch. In **Settings → Pages**, select **Deploy from a branch**, **main**, **/docs**, then Save. The precomputed explorer requires no server-side Python on Pages. Keep patient data, clinical weights and local update files outside version control.
