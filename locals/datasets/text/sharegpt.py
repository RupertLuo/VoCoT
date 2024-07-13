from locals.datasets.text.text_data_base import TextDataset

class ShareGPTDataset(TextDataset):
    def __init__(self, 
        path: str, 
        instruct: bool = False, 
        sample_weight: float = 1, 
        output_mode: str = 'conversation', 
        shuffle: bool = False, 
        inference: bool = False, 
        **kwargs):
        path = [path]
        super().__init__(path, instruct, sample_weight, output_mode, shuffle, inference, **kwargs)
        print(f"ShareGPTDataset has {len(self)} samples!!")

    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        i = self.get_sampler_index(i)
        item = self.meta[i]
        return {'conversation': item['conversations'],'id':item['id']}

class ShareGPTCodeDataset(TextDataset):
    def __init__(self, 
        path: str, 
        instruct: bool = False, 
        sample_weight: float = 1, 
        output_mode: str = 'conversation', 
        shuffle: bool = False, 
        inference: bool = False, 
        **kwargs):
        path = [path]
        super().__init__(path, instruct, sample_weight, output_mode, shuffle, inference, **kwargs)
        print(f"ShareGPTCodeDataset has {len(self)} samples!!")

    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        i = self.get_sampler_index(i)
        item = self.meta[i]
        return {'conversation': item['conversations'],'id':'sharegpt_code_'+item['id']}
    


if __name__ == '__main__':
    test_path = '/mnt/bn/yangmin-priv/luoruipu/data/text-dataset/sharegpt_184k_no_repeat.json'
    test = ShareGPTDataset(path=test_path)
    print(len(test))
    print(test[100000])