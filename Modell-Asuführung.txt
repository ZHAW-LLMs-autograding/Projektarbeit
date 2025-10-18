----------------------------------------------------------------------------------------------------

Hello-AI

Environment auswählen
source ~/projektarbeit/asag_finetune/legacy_import/hello-ki/bert_lab/.hello_bert/bin/activate

Bias-Only: Ins Verzeichnis wechseln
cd ~/projektarbeit/asag_finetune/legacy_import/hello-ki/bert_lab/BURT/ayser_bias_only

Skript ausführen
python -u talk_with_bert_mask_ayser.py

Eingabe Testen
The sky is [MASK].
A house is [MASK].
A university [MASK] a building.
[MASK] house is red.
The house [MASK] windows.

-------------------------------------------------------

Head-Only

Ins Verzeichnis wechseln
cd ~/projektarbeit/asag_finetune/legacy_import/hello-ki/bert_lab/BURT/ayser_head_only/

Skript ausführen
python -u talk_with_bert_mask_ayser.py

Eingabe Testen
The sky is [MASK].
A house is [MASK].
A university [MASK] a building.
[MASK] house is red.
The house [MASK] windows.

----------------------------------------------------------------------------------------------------

Basis Bert ausführen

Environment auswählen
source ~/projektarbeit/asag_finetune/env/bin/activate

Test Skript ausführen
python ~/projektarbeit/asag_finetune/scripts/bert_mlm_repl.py

Eingabe Testen
The sky is [MASK].
A house is [MASK].
A university [MASK] a building.
[MASK] house is red.
The house [MASK] windows.

----------------------------------------------------------------------------------------------------

Bert mit Sentence Transformer

Environment auswählen
source ~/projektarbeit/asag_finetune/env/bin/activate

Test Skript ausführen
python ~/projektarbeit/asag_finetune/scripts/bert_st_repl.py

Eingabe Testen
The sky is [MASK].
A house is [MASK].
A university [MASK] a building.
[MASK] house is red.
The house [MASK] windows.

----------------------------------------------------------------------------------------------------

BERT Auswertungs Modell

Environment auswählen
source ~/projektarbeit/asag_finetune/env/bin/activate

Test Skript mit eigenem benchmark ausführen ausführen
python ~/projektarbeit/asag_finetune/scripts/test_cases_ordinal.py

Eingabe testen
python ~/projektarbeit/asag_finetune/scripts/infer_bert_asag_ordinal_ctx.py --model_dir ~/projektarbeit/asag_finetune/experiments/bert_asag_v1/ckpts_student_ord_ctx/best --question  "Why does increasing resistance decrease the current in a circuit?" --reference "Because I = V/R; higher resistance reduces current at fixed voltage." --answer "Because higher resistance reduces current at fixed voltage."

----------------------------------------------------------------------------------------------------

Bert Test starten mit eigenen Benchmark

Environment wählen
source ~/projektarbeit/asag_finetune/env/bin/activate

Test laufen lassen
python ~/projektarbeit/asag_finetune/scripts/run_benchmark5class.py

Resultate
ls -lh ~/projektarbeit/asag_finetune/experiments/bert_asag_v1/runs/benchmark5class_results.csv
ls -lh ~/projektarbeit/asag_finetune/experiments/bert_asag_v1/runs/metrics_summary_5class.csv

Ausführung anzeigen
cat ~/projektarbeit/asag_finetune/experiments/bert_asag_v1/runs/benchmark5class_results.csv
Resultate anzeigen
cat ~/projektarbeit/asag_finetune/experiments/bert_asag_v1/runs/metrics_summary_5class.csv

----------------------------------------------------------------------------------------------------

Llama 3.1 starten

Environment starten
source ~/projektarbeit/inference/inference_venv/bin/activate

Llama Server starten
python -m vllm.entrypoints.openai.api_server \
  --model "Undi95/Meta-Llama-3.1-8B-Claude" \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
  
In einer neuen Shell Llama starten mit
~/projektarbeit/inference/query_claude_3.1.sh

Training starten:
python3 ~/projektarbeit/train_llama31.py --config ~/projektarbeit/llama31_config.json

config-file bearbeiten
nano ~/projektarbeit/llama31_config.json

----------------------------------------------------------------------------------------------------
