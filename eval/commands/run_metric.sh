##################### total setting=2,4,6,7
function run_all(){
    model_name=/mnt/bn/yangmin-priv/luoruipu/checkpoints/LLaVA-clip336px-obj-represent-Mistral-1e-5-3072-instruct_llava+shikraCoT+GPTQTA-Block+lvis-cot/
    
    
    store_model_name=llava_mistral_instruct_simplified_image_llava_shikraCoT75+GPTQTA-qa2t

    ## dataset setting
    function mmbench(){
        ##### mci
        dataset_name=mmbench
        dataset_config=config/datasets/eval/MMBench_DEV_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        echo "========MMBench Result=========="
        python3 eval/eval_tools/mmbench.py --config  ${dataset_config}  --result  ${output_dir}/MMBench_DEV_opt.json
    }

    function seed(){
        ##### mci
        dataset_name=seed
        dataset_config=config/datasets/eval/SEED_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        echo "========SEED Result=========="
        python3 eval/eval_tools/seed.py --config ${dataset_config}  --result  ${output_dir}/SEED_opt.json
    }

    function clevr(){
        ##### mci
        dataset_name=clevr
        dataset_config=config/datasets/eval/CLEVR_val1k_type_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        echo "========CLEVR Result=========="
        python3 eval/eval_tools/clevr.py --config config/datasets/eval/CLEVR_val1k_type_opt.yaml --result  ${output_dir}/CLEVR_val1k_type_opt.json
    }


    function embspatial(){
        ##### mci
        dataset_name=emb_spa
        dataset_config=config/datasets/eval/EmbSpa_test_opt.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        echo "========EmbSpatial Result==========="
        python3 eval/eval_tools/mp3d.py --config config/datasets/eval/EmbSpa_test_opt.yaml --result ${output_dir}/EmbSpa_test_opt.json
    }

    function vsr(){
        ##### mci
        dataset_name=vsr
        dataset_config=config/datasets/eval/VSR_TEST.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        echo "========VSR Result==========="
        python3 eval/eval_tools/vsr.py --config config/datasets/eval/VSR_TEST.yaml  --data ${output_dir}/VSR_TEST.json
    }
    
    function pope(){
        ##### mci
        dataset_name=pope
        dataset_config=config/datasets/eval/POPE_adversarial.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        echo "========POPE Result==========="
        python3 eval/eval_tools/pope.py --data  ${output_dir}/POPE_adversarial.json
    }

    function vstar(){
        ##### mci
        dataset_name=vstar
        dataset_config=config/datasets/eval/VStar.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        echo "========VSTAR Result==========="
        python3 eval/eval_tools/vstar.py --result  ${output_dir}/VStar.json
    }

    function wino(){
        ##### mci
        dataset_name=wino
        dataset_config=config/datasets/eval/Wino.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_no_instruct/
        echo "========WINO-txt Result==========="
        python3 eval/eval_tools/vstar.py --result  ${output_dir}/Wino.json
    }

    function amber(){
        dataset_name=amber
        dataset_config=config/datasets/eval/AMBER_desc.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot_sum/
        echo "========AMBER Result Need Further Evaluation==========="
        python3 eval/eval_tools/convert_res_to_amber.py \
          --src ${output_dir}/AMBER_desc.json \
          --tgt ${output_dir}/AMBER_desc_eval.json \
          --desc
    }

    function gqa(){
        dataset_name=gqa
        dataset_config=config/datasets/eval/GQA.yaml
        output_dir=output/${dataset_name}/${store_model_name}/cot/
        echo "========GQA Result Need Further Evaluation==========="
        python3 eval/eval_tools/convert_res_to_gqa.py \
          --src ${output_dir}/GQA.json \
          --dst ${output_dir}/testdev_balanced_predictions.json
    }

    function refcocog_test(){
        ##### mci
        dataset_name=refcoco
        dataset_config=config/datasets/eval/refcocog_test-u.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        echo "========RefCOCOg test Result==========="
        python3 eval/eval_tools/refcoco.py --mistral  --path ${output_dir}/refcocog_test-u.json
    }
    function refcocog_val(){
        ##### mci
        dataset_name=refcoco
        dataset_config=config/datasets/eval/refcocog_val-u.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        echo "========RefCOCOg val Result==========="
        python3 eval/eval_tools/refcoco.py --mistral  --path ${output_dir}/refcocog_val-u.json
    }
    function clevr_ref(){
        dataset_name=refcoco
        dataset_config=config/datasets/eval/clevr_ref.yaml
        output_dir=output/${dataset_name}/${store_model_name}/llava_prompt/
        echo "========CLEVR REF Result==========="
        python3 eval/eval_tools/refcoco.py --mistral  --path ${output_dir}/clevr_ref.json
    }

mmbench
seed
embspatial
clevr
wino
vsr
pope
vstar
refcocog_val
refcocog_test
clevr_ref
gqa
amber
}
run_all