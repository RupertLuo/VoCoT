from locals.datasets.text.text_data_base import TextDataset
from pathlib import Path
class UltraChatDataset(TextDataset):
    def __init__(self, 
        path: str, 
        instruct: bool = False, 
        sample_weight: float = 1, 
        output_mode: str = 'conversation', 
        shuffle: bool = False, 
        inference: bool = False, 
        **kwargs):
        path = [str(p)for p in Path(path).glob('*.jsonl')]
        super().__init__(path, instruct, sample_weight, output_mode, shuffle, inference, **kwargs)
        self.filter_dataset()
        print(f"UltraChat Dataset has {len(self)} samples!!")

    def filter_dataset(self,):
        new_data = []
        for item in self.meta:
            try:
                assert len(item['data'])%2==0
                new_data.append(item)
            except:
                continue
        self.meta = new_data

    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        i = self.get_sampler_index(i)
        item = self.meta[i]
        conversations = []
        for i in range(0,len(item['data']),2):
            conversations.append({'from':'human','value':item['data'][i]})
            conversations.append({'from':'gpt','value':item['data'][i+1]})
        return {'conversation': conversations,'id':'UltraChat'+str(item['id'])}

class UltraChatJsonDataset(TextDataset):
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
        self.filter_dataset()
        print(f"UltraChat Dataset has {len(self)} samples!!")


    def filter_dataset(self,):
        new_data = []
        length_list = []
        for item in self.meta:
            try:
                assert len(item['conversations'])%2==0 and len(item['conversations'])!=0
                new_data.append(item)
            except:
                continue
            finally:
                cur_len = sum(len(conv['value'].split()) for conv in item['conversations'])
                cur_len = cur_len if 'image' in item else -cur_len
                length_list.append(cur_len)
        self.meta = new_data
        self.length = length_list

    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        item = self.meta[i]
        return {'conversation': item['conversations'],'id':'UltraChat'+str(item['id'])}


if __name__ == '__main__':
    ultrachat_path = '/mnt/bn/yangmin-priv/luoruipu/data/text-dataset/ultrachat/'
    test = UltraChatDataset(path=ultrachat_path)
    print(len(test))
    print(test[0])