from matplotlib import pyplot as plt
import numpy as np

class graphPlot:
    def __init__(self, mapname):
        pass

    def plot_all(self):
        map_name_list = ["gbr","esp","mco"]
        testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering","control_delay_speed","control_Delay_steering","perception_delay"]
    
        for mapname in map_name_list:
            self.planned_path = np.loadtxt(f"./maps/{mapname}_raceline.csv", delimiter=",")
            
            for testmode in testmode_list:

                if self.TESTMODE == "Benchmark" or self.TESTMODE == " ": 
                    self.Max_iter=7           
                elif self.TESTMODE == "perception_noise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
                    self.Max_iter = 30
                elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_Delay_steering" or self.TESTMODE == "perception_delay":
                    self.Max_iter = 10

                for iter in range(self.Max_iter):
                    self.trajectory_plot(mapname,testmode,iter)
                

    def trajectory_plot(self, mapname, testmode, iter):
        # laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error
        data = np.loadtxt(f"./Imgs/{mapname}/{testmode}/{iter}.csv", delimiter=",")
        plt.figure()
        plt.title("Planned vs Car trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(data[:,1], data[:,2],label = "Car Trajectory")
        plt.plot(self.planned_path[:,1],self.planned_path[:,2], label = "Planned Trajectory")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./Imgs/{mapname}/{testmode}/{iter}.svg")
        plt.clf()



def main():
    pass

if __name__ == "__main__":
    main()