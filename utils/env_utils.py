"""
Cart Pole environment input extraction
Code adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from gym import wrappers
import os
resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Grayscale(),
                             transforms.Resize(40, interpolation=Image.CUBIC),
                             transforms.ToTensor()])


def get_cart_location(env,screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
    # 이것을 Torch order (CHW)로 변환한다.
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 카트는 아래쪽에 있으므로 화면의 상단과 하단을 제거하십시오.
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env,screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 카트를 중심으로 정사각형 이미지가 되도록 가장자리를 제거하십시오.
    screen = screen[:, :, slice_range]
    # float 으로 변환하고,  rescale 하고, torch tensor 로 변환하십시오.
    # (이것은 복사를 필요로하지 않습니다)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
   # screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
    #screen=np.expand_dims(screen, axis=0)
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
    return resize(screen)

def make_video(env, model,device):
    try:
        os.makedirs(os.path.join(os.getcwd(), "videos"))
    except:
        print("Path Already Exists")
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        action = model.sample_action(torch.tensor(observation).unsqueeze(0).float().to(device), 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


















from itertools import chain 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np






def render(env):
    
    #Print 
    def index_fun(a,b,height):
        return a*height+b
    def state_to_color(argument):
        func=switcher.get(argument,"nothing")
        return func()
    def frozen():
        return 'c'
    def hole():
        return 'k'
    def start():
        return "y"
    def goal():
        return 'g'
    switcher = {
        'F': frozen,
        'S': start,
        'H': hole,
        'G': goal
    }
    desc = env.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    desc_flatten = list(chain.from_iterable(desc)) 
    height=len(desc)
    width=len(desc[0])
    fig=plt.figure(figsize=(4, 4))

    ax = [plt.subplot(height,width,i+1) for i in range(height*width)]
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.setp(ax, xlim=(-1,1), ylim=(-1,1))



    def state_to_color(argument):
        func=switcher.get(argument,"nothing")
        return func()
    for index,(axes,state) in enumerate(zip(ax,desc_flatten)):

        axes.set_facecolor(state_to_color(state))
        if index==env.s:
            axes.set_facecolor((1,0,0))

        axes.get_yaxis().set_visible(False)
        axes.get_xaxis().set_visible(False)


    plt.subplots_adjust(wspace=0, hspace=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img