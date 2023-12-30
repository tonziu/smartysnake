import pygame as pg
import numpy  as np

pg.init()


class Snake:
    def __init__(self, x0, y0, width, height, color):
        self.body = [pg.Rect(x0, y0, width, height)]
        self.color = color
        self.dir = [0, 0]
        self.moves = 50

    def render(self, surface):
        for rect in self.body:
            pg.draw.rect(surface, self.color, rect, 2, 3)

    def steer_left(self):
        if self.dir[0] != 1:
            self.dir = [-1, 0]
            self.moves -= 1

    def steer_right(self):
        if self.dir[0] != -1:
            self.dir = [1, 0]
            self.moves -= 1

    def steer_up(self):
        if self.dir[1] != 1:
            self.dir = [0, -1]
            self.moves -= 1

    def steer_down(self):
        if self.dir[1] != -1:
            self.dir = [0, 1]
            self.moves -= 1

    def move(self):
        for i in range(len(self.body)-1, 0, -1):
            self.body[i].x = self.body[i-1].x
            self.body[i].y = self.body[i-1].y

        self.head.x += (self.dir[0] * self.width)
        self.head.y += (self.dir[1] * self.height)

    def grow(self):
        new_head = pg.Rect(self.head.x, self.head.y, self.width, self.height)
        self.body.append(new_head)
        self.moves += 50

    @property
    def head(self):
        return self.body[0]

    @property
    def tail(self):
        return self.body[-1]

    @property
    def width(self):
        return self.head.width

    @property
    def height(self):
        return self.head.height

    def __len__(self):
        return len(self.body)


class Game:
    def __init__(self):
        self.screen_width = 0
        self.screen_height = 0
        self.running = 0
        self.fps = 0
        self.bg_color = '0x000000'
        self.wall_color = '0x000000'
        self.snake_color = '0x000000'
        self.food_color = '0x000000'
        self.cell_size = 0
        self.num_cells = 0
        self.walls = []
        self.fitness = 0

        self.screen = None
        self.clock = None
        self.snake = None
        self.food = None

    def quit(self):
        self.running = 0
        self.wall_color = '0x860000'

    def pick_random_cell(self):
        x = np.random.randint(2, self.num_cells - 2) * self.cell_size
        y = np.random.randint(2, self.num_cells - 2) * self.cell_size
        return x, y

    def pick_free_cell(self):
        done = False
        x, y = self.pick_random_cell()
        while not done:
            done = True
            for rect in self.snake.body:
                if x == rect.x and y == rect.y:
                    done = False
            if not done:
                x, y = self.pick_random_cell()
        return x, y

    def init_screen(self):
        return pg.display.set_mode((self.screen_width, self.screen_height))

    def init_walls(self):
        walls = []
        for i in range(self.num_cells):
            top_rect = pg.Rect(i*self.cell_size, 0, self.cell_size, self.cell_size)
            right_rect = pg.Rect(self.screen_width-self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            bottom_rect = pg.Rect(i*self.cell_size, self.screen_height-self.cell_size, self.cell_size, self.cell_size)
            left_rect = pg.Rect(0, i * self.cell_size, self.cell_size, self.cell_size)
            walls.append(top_rect)
            walls.append(right_rect)
            walls.append(bottom_rect)
            walls.append(left_rect)
        return walls

    def init_food(self):
        x, y = self.pick_free_cell()
        return pg.Rect(x, y, self.cell_size, self.cell_size)

    def init_snake(self):
        x,y = self.pick_random_cell()
        return Snake(x, y, self.cell_size, self.cell_size, self.snake_color)

    def startup(self, rendering = True):
        self.wall_color = '0x128686'
        if rendering:
            self.screen = self.init_screen()
            self.clock = self.init_clock()
        self.running = 1
        self.walls = self.init_walls()
        self.snake = self.init_snake()
        self.food = self.init_food()
        self.fitness = 0

    def dispatch_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    self.snake.steer_left()
                elif event.key == pg.K_UP:
                    self.snake.steer_up()
                elif event.key == pg.K_RIGHT:
                    self.snake.steer_right()
                elif event.key == pg.K_DOWN:
                    self.snake.steer_down()

    def update_ai(self, network):
        network_out = network.feed(self.state)
        decision = np.argmax(network_out)
        if decision == 0:
            self.snake.steer_up()
        elif decision == 1:
            self.snake.steer_right()
        elif decision == 2:
            self.snake.steer_down()
        elif decision == 3:
            self.snake.steer_left()

    def update(self):
        self.snake.move()

        # WALL COLLISION CHECK
        if self.snake.head.x < self.cell_size:
            self.quit()
        elif self.snake.head.y < self.cell_size:
            self.quit()
        elif self.snake.head.right > self.screen_width - self.cell_size:
            self.quit()
        elif self.snake.head.bottom > self.screen_height - self.cell_size:
            self.quit()

        # FOOD COLLISION CHECK
        if self.snake.head.x == self.food.x and self.snake.head.y == self.food.y:
            self.snake.grow()
            self.food = self.init_food()
            self.fitness += 10 * (len(self.snake)**2)

        # BODY COLLISION CHECK

        for rect in self.snake.body[1:-1]:
            if rect.x == self.snake.head.x and rect.y == self.snake.head.y:
                self.quit()
                break

        # MOVES COUNT CHECK

        if self.snake.moves <= 0:
            self.quit()

    def render_walls(self):
        for rect in self.walls:
            pg.draw.rect(self.screen, self.wall_color, rect, 2, 3)

    def render_food(self):
        pg.draw.rect(self.screen, self.food_color, self.food, 2, 3)

    def render(self):
        self.screen.fill(self.bg_color)
        self.render_walls()
        self.render_food()
        self.snake.render(self.screen)
        pg.display.update()

    def test(self, rendering=True):
        self.startup(rendering)
        while self.running:
            if rendering:
                self.dispatch_events()
            self.update()
            if rendering:
                self.render()
                self.clock.tick(self.fps)
            print(self.fitness)
        self.quit()
        return self.fitness

    def showcase(self, network):
        self.startup(rendering=True)
        while self.running:
            self.update_ai(network)
            self.update()
            self.render()
            self.clock.tick(self.fps)
        self.quit()

    def play_ai(self, network):
        self.startup(rendering=False)
        while self.running:
            prev_dist_to_food = np.sqrt((self.snake.head.x-self.food.x)**2 + (self.snake.head.y-self.food.y)**2)
            self.update_ai(network)
            self.update()
            curr_dist_to_food = np.sqrt((self.snake.head.x - self.food.x) ** 2 + (self.snake.head.y - self.food.y) ** 2)
            if curr_dist_to_food < prev_dist_to_food:
                self.fitness += 0.2
            else:
                self.fitness -= 0.1
        return self.fitness

    @property
    def state(self):
        # distance to wall
        x1 = (self.snake.head.x - self.cell_size) / (self.screen_width - 2*self.cell_size)
        x2 = (self.snake.head.y - self.cell_size) / (self.screen_width - 2*self.cell_size)
        x3 = ((self.screen_width - self.cell_size)-self.snake.head.x) / (self.screen_width - 2*self.cell_size)
        x4 = ((self.screen_height - self.cell_size) - self.snake.head.y) / (self.screen_width - 2*self.cell_size)
        # distance to food
        x5 = self.food.x / (self.screen_width - 2 * self.cell_size)
        x6 = self.food.y / (self.screen_width - 2 * self.cell_size)
        x7 = ((self.screen_width - self.cell_size) - self.food.x) / (self.screen_width - 2 * self.cell_size)
        x8 = ((self.screen_height - self.cell_size) - self.food.y) / (self.screen_width - 2 * self.cell_size)
        # distance to tail
        x9 = self.snake.tail.x / (self.screen_width - 2 * self.cell_size)
        x10 = self.snake.tail.y / (self.screen_width - 2 * self.cell_size)
        x11 = ((self.screen_width - self.cell_size) - self.snake.tail.x) / (self.screen_width - 2 * self.cell_size)
        x12 = ((self.screen_height - self.cell_size) - self.snake.tail.y) / (self.screen_width - 2 * self.cell_size)
        # direction
        x13 = 1 if (self.snake.dir[0] == 1) else 0
        x14 = 1 if (self.snake.dir[0] == -1) else 0
        x15 = 1 if (self.snake.dir[1] == 1) else 0
        x16 = 1 if (self.snake.dir[1] == -1) else 0
        return [[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]]

    @staticmethod
    def init_clock():
        return pg.time.Clock()

    @staticmethod
    def default():
        new_game = Game()
        new_game.screen_width = 400
        new_game.screen_height = 400
        new_game.fps = 25
        new_game.bg_color = '0x252525'
        new_game.wall_color = '0x128686'
        new_game.snake_color = '0x128612'
        new_game.food_color = '0xaa1212'
        new_game.num_cells = 25
        new_game.cell_size = new_game.screen_width // new_game.num_cells
        return new_game


if __name__ == "__main__":
    test_game = Game.default()
    test_game.test()