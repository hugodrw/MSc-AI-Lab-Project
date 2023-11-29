import pygame
from Planet import planet
import numpy as np
from Extra import *
from environment import BaseEnvironment

WIDTH, HEIGHT = 1000, 800
EARTH_RADIUS = 100
SATELLITE_RADIUS = 7
WHITE = (255, 255, 255)
TIME_STEP = 4

ep_count = "None Found"

VISUALISE = True # add a visualise toggle bool
REACHED_DIST = 10



class SatelliteEnvironment(BaseEnvironment):
    def __init__(self):
        self.name = "Satellite Simulator"
        

    def env_init(self , env_info={}):
        if VISUALISE:
            # --- Pygame initialisation --- #
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Satellite Simulator")
            self.clock = pygame.time.Clock()
            # --- Pygame initialisation --- #


        # --- Objects creation --- #
        self.earth = planet(mass=1000 ,
                    name="Earth" ,
                    position=np.array([WIDTH // 2, HEIGHT // 2]))
        self.satellite_1 = planet(mass=1,
                            name="Satellite_1",
                            position=np.array([self.earth.position[0] + EARTH_RADIUS + 60, self.earth.position[1]]))
        self.satellite_1.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=EARTH_RADIUS + 60)


        sat_2_init_pos = np.array(polar_to_cartesian(-90 , EARTH_RADIUS+140))
        sat_2_init_pos += self.earth.position

        self.satellite_2 = planet(mass=1,
                            name="Satellite_2",
                            position=sat_2_init_pos)
        self.satellite_2.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=EARTH_RADIUS + 140)
        
        self.pl_array = [self.earth]
        # --- Objects creation --- #

        # --- Other values --- #
        self.sat_1_fuel = 100
        self.initial_distance = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        

        observation = self.env_observe_state()
        self.last_observation = observation
        return observation


    

    def values_update(self):
        self.satellite_1.update_velocity(self.pl_array , dt=TIME_STEP)
        self.satellite_1.update_pos(dt=TIME_STEP)
        self.satellite_2.update_velocity(self.pl_array , dt=TIME_STEP)
        self.satellite_2.update_pos(dt=TIME_STEP)

    def visual_update(self):
        if VISUALISE == False:
            return

        screen = self.screen
        earth = self.earth
        satellite_1 = self.satellite_1
        satellite_2 = self.satellite_2
        screen.fill((3, 9, 41))

        # orbits
        pygame.draw.circle(screen, (135, 135, 135), earth.position, EARTH_RADIUS + 60, 1)
        pygame.draw.circle(screen, (135, 135, 135), earth.position, EARTH_RADIUS + 140, 1)

        # earth
        pygame.draw.circle(screen, (4, 113, 135), earth.position, EARTH_RADIUS)  # Fill
        pygame.draw.circle(screen, WHITE, earth.position, EARTH_RADIUS, 1)       # Border

        # satellite
        line(satellite_1.position , satellite_2.position , screen , (51, 77, 47))
        pygame.draw.circle(screen, WHITE, (int(satellite_1.position[0]),
                                           int(satellite_1.position[1])),
                                           SATELLITE_RADIUS)
        line(satellite_1.position , satellite_1.position + satellite_1.normalise_vector(satellite_1.velocity)*60 , screen , WHITE)
        
        pygame.draw.circle(screen, WHITE, (int(satellite_2.position[0]),
                                           int(satellite_2.position[1])),
                                           SATELLITE_RADIUS)
        line(satellite_2.position , satellite_2.position + satellite_2.normalise_vector(satellite_2.velocity)*60 , screen , WHITE)
        

        # Display satellite velocity and altitude
        label(f"Satellite 1" , (WIDTH - 300, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_1.velocity), 2)}" , (WIDTH - 300, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_1.position - earth.position) - EARTH_RADIUS, 2)}" , (WIDTH - 300, 100) , screen )
        label(f"Satellite 2" , (WIDTH - 170, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_2.velocity), 2)}" , (WIDTH - 170, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_2.position - earth.position) - EARTH_RADIUS, 2)}" , (WIDTH - 170, 100) , screen )
        label(f"Distance: {round(np.linalg.norm(satellite_1.position - satellite_2.position) , 2)}" , (WIDTH - 170, 140) , screen )
        label("Orbit 1" , (WIDTH - 365, HEIGHT//2 + 80) , screen)
        label("Orbit 2" , (WIDTH - 290, HEIGHT//2 + 110) , screen)
        
        label(ep_count , (20, 20) , screen)
        label(f"Fuel: {self.sat_1_fuel}" , (WIDTH - 300 , 140) , screen)
            
        pygame.display.flip()
        self.clock.tick(60)


    def env_observe_state(self):
        # sat_1 altitude
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - EARTH_RADIUS
        # dist(sat_1 , sat_2)
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        # sat_1 fuel left
        fuel = self.sat_1_fuel

        
        #print("OBSERVING :" , (sat_1_alt , dist , fuel))
        return (sat_1_alt , dist , fuel)


    def perform_action(self , a):
        # Observe current state
        current_state = self.env_observe_state()

        # Perform action
        if a == 1:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , 0.01)
        elif a == 2:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , -0.01)
        elif a == 4:
            # 4: set velocity to stay in orbit
            self.satellite_1.set_circular_orbit_velocity(self.earth , self.satellite_1.calculate_distance(self.satellite_1.position , self.earth.position))

        # Observe new state
        next_state = self.env_observe_state()

        # Calculate reward
        reward = self.calculate_reward(current_state, a, next_state)

        is_terminal = self.is_terminal(next_state)
        return (reward, next_state, is_terminal)
        
        
    def define_possible_actions(self):
        # Actions: 1: accelerate, 2: decelerate, 3: wait
        return [1,2,3]

    def is_terminal(self , state):
        sat_1_alt , dist , fuel = state

        if sat_1_alt < 0: # Satellite has crashed on Earth
            return True
        elif dist < REACHED_DIST: # Satellite has reached objective
            return True
        elif fuel <= 0: # Satellite has no more fuel
            return True
        elif dist > 300: # Satellite goes too far
            return True
        
        return False 
    

    def calculate_reward(self , state , action , next_state):
        sat_1_alt , dist , fuel = state
        next_sat_1_alt , next_dist , next_fuel = next_state
        reward = 0

        if action == 1 or action == 2: # Using fuel
            reward -= 1

        if dist > REACHED_DIST and next_dist <= REACHED_DIST: # Reaching objective
            reward += 10000

        if sat_1_alt > 0 and next_sat_1_alt <= 0: # Crashing on Earth
            reward -= 1000

        if fuel <= 0 and dist > REACHED_DIST: # Fail to reach objective
            reward -= 1000

        return reward

    

    def env_start(self):
        reward = 0.0
        
        is_terminal = False

        self.values_update()
        self.visual_update()

        observation = self.env_init()
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term


    def env_step(self, action):
        # Take a step in the environment based on the given action
        # Perform the action in your environment
        self.perform_action(action)

        

        self.values_update()
        self.visual_update()

        # Observe the new state
        next_state = self.env_observe_state()

        # Check if the episode is terminal
        is_terminal = self.is_terminal(next_state)

        # Calculate the reward for the current state, action, and next state
        reward = self.calculate_reward(self.last_observation, action, next_state)

        # Update the last observation
        self.last_observation = next_state

        # Return the tuple (reward, next_state, is_terminal)
        return (reward, next_state, is_terminal)

    def env_end(self):
        # End the current episode
        pass

    def env_cleanup(self):
        # Clean up the environment
        self.env_init()

    def pass_count(self , message):
        global ep_count
        ep_count = message
        
