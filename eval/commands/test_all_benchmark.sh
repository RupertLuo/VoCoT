##################### total setting=2,4,6,7
function run_all(){
    model_name=PATH/TO/YOUR_MODEL
    
    
    store_model_name=YOUR_MODEL_NAME

    ## dataset setting
    function mmbench(){
        ##### mci
        dataset_name=mmbench
        dataset_config=config/datasets/eval/MMBench_DEV_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot   
            --evaluate_loss 
            --max_new_tokens 2048  
            --option_instruct
            --likelihood_reduction sum
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function seed(){
        ##### mci
        dataset_name=seed
        dataset_config=config/datasets/eval/SEED_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot  
            --evaluate_loss 
            --max_new_tokens 2048  
            --option_instruct
            --likelihood_reduction sum
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function clevr(){
        ##### mci
        dataset_name=clevr
        dataset_config=config/datasets/eval/CLEVR_val1k_type_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot
            --evaluate_loss 
            --max_new_tokens 2048  
            --option_instruct
            --likelihood_reduction sum
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function embspatial(){
        ##### mci
        dataset_name=emb_spa
        dataset_config=config/datasets/eval/EmbSpa_test_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot  
            --evaluate_loss 
            --max_new_tokens 2048  
            --option_instruct
            --likelihood_reduction sum
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function vsr(){
        ##### mci
        dataset_name=vsr
        dataset_config=config/datasets/eval/VSR_TEST.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot    
            --max_new_tokens 2048  
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
    
    function pope(){
        ##### mci
        dataset_name=pope
        dataset_config=config/datasets/eval/POPE_adversarial.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot   
            --max_new_tokens 2048  
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function gqa(){
        ##### mci
        dataset_name=gqa
        dataset_config=config/datasets/eval/GQA.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot   
            --max_new_tokens 2048  
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }


    function vstar(){
        ##### mci
        dataset_name=vstar
        dataset_config=config/datasets/eval/VStar.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot   
            --evaluate_loss 
            --max_new_tokens 2048  
            --option_instruct
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }

    function wino(){
        ##### mci
        dataset_name=wino
        dataset_config=config/datasets/eval/Wino.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_no_instruct/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot  
            --evaluate_loss 
            --max_new_tokens 2048  
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
    
    function amber(){
        ##### mci
        dataset_name=amber
        dataset_config=config/datasets/eval/AMBER_desc.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --cot   --desc 
            --max_new_tokens 2048  
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
    
    function refcocog_test(){
        ##### mci
        dataset_name=refcoco
        dataset_config=config/datasets/eval/refcocog_test-u.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --max_new_tokens 2048
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
    function refcocog_val(){
        ##### mci
        dataset_name=refcoco
        dataset_config=config/datasets/eval/refcocog_val-u.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --max_new_tokens 2048
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
    function clevr_ref(){
        ##### mci
        dataset_name=refcoco
        dataset_config=config/datasets/eval/clevr_ref.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        #output_dir=output/mfdu_output/mci_output/${store_model_name}_${model_name}_${infer_method}_${formulation}
        # --model_type ${model_type1}
        flag=" --model_path ${model_name}
            --eval_data ${dataset_config} 
            --output_dir ${output_dir}
            --avoid_image_gen 
            --temperature 0 
            --precision fp16 
            --expand2square 
            --use_mistral 
            --max_new_tokens 2048
            "
        torchrun --nproc_per_node=8 --master_port=8801 eval/evaluate_benchmark.py $flag
    }
wino
pope
vstar
vsr
mmbench
seed
embspatial
clevr
gqa
amber
clevr_ref
refcocog_test
refcocog_val
}
run_all