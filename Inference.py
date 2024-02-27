from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')


def run(prompt, output_video_path):
    
    test_text = {
            'text':prompt,
        }
    output_video_path = p(test_text, output_video=output_video_path)[OutputKeys.OUTPUT_VIDEO]
    print('output_video_path:', output_video_path)
    
if __name__ == '__main__':
    prompt = 'A cat eating food out of a bowl,in style of van Gogh.'
    run(prompt, 'output_video.mp4')