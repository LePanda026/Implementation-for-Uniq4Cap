import json
import os
from pydub import AudioSegment

config_path = '/home/w/LanguageBind/encoder/video_config.json'
data_path='/home/w/LanguageBind/coco_vat_test/'
end_audio='.m4a'
def convert_m4a_to_wav(m4a_path):
    # 检查传入的文件是否是M4A格式
    if not m4a_path.endswith('.m4a'):
        print("文件不是M4A格式，无法转换。")
        return

    # 构建WAV文件的完整路径
    wav_file_name = os.path.splitext(os.path.basename(m4a_path))[0] + '.wav'
    wav_path = os.path.join(os.path.dirname(m4a_path), wav_file_name)

    # 转换M4A到WAV
    sound = AudioSegment.from_file(m4a_path, format="m4a")
    sound.export(wav_path, format="wav")

    # 删除源M4A文件
    os.remove(m4a_path)

    print(f"已将 {m4a_path} 转换为 {wav_path} 并删除源文件。")

# # 使用示例
# m4a_file_path = '/home/w/LanguageBind/coco_vat_test/zk74BK7wJqA.m4a'
# convert_m4a_to_wav(m4a_file_path)
with open(config_path, 'r') as file:
    with open('/home/w/LanguageBind/coco_vat_test.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 循环取出所有的键
    for key in data.keys():
        audio_path = os.path.join(data_path, key + end_audio)
        convert_m4a_to_wav(audio_path)
        # sound_mplug_value = data.get(key, {}).get("mplug")
        # print(sound_mplug_value)




