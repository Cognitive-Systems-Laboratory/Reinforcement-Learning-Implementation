"""
Cart Pole environment input extraction
Code adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(40, interpolation=Image.CUBIC),
                             transforms.ToTensor()])
class CartPoleEnv:
    def __init__(self, screen_width):
        super().__init__()
        self.screen_width = screen_width

    def get_cart_location(self, env):
        world_width = env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self, env):
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location(env)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0)
    
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