export OPENAI_API_BASE="https://api.lingyiwanwu.com/v1"
export OPENAI_API_KEY="alignment-eval"
export CHAT_MODEL_NAME="gpt-4o-global"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

HF_MODEL_ID="/gpfs/public/infra/zyk/20240904/exp35a_gpt4orm_offinit_data05054oref_numres_8_buffer512_gbs512_lr2e-54e-32e-4constant_ipo_seqlen1k_fixedgap2stratified_nocap_run5000/cvt_ckpts/iter_0000064-hf/" # example model id 
MODEL_PRETTY_NAME="exp35a_gpt4orm_offinit_data05054oref_numres_8_buffer512_gbs512_lr2e-54e-32e-4constant_ipo_seqlen1k_fixedgap2stratified_nocap_run5000_iter64" # example model name
NUM_GPUS=8 # depending on your hardwares;
# do inference on WildBench 
bash scripts/_common_vllm.sh $HF_MODEL_ID $MODEL_PRETTY_NAME $NUM_GPUS 
# submit to OpenAI for eval (WB-Score)
bash evaluation/run_score_eval_batch.sh ${MODEL_PRETTY_NAME} 
# check the batch job status
python src/openai_batch_eval/check_batch_status_with_model_name.py ${MODEL_PRETTY_NAME} 
# show the table 
bash leaderboard/show_eval.sh score_only 
