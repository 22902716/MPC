from matplotlib import pyplot as plt
import numpy as np

class graphPlot:
    def __init__(self):
        self.control_method = ["PurePursuit", "MPC"]
        self.map_name_list = ["gbr","esp","mco"]
        self.testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering","control_delay_speed","control_Delay_steering","perception_delay"]

    def plot_all(self):

        plt.figure()
        for mapname in self.map_name_list:
            self.planned_path = np.loadtxt(f"./maps/{mapname}_raceline.csv", delimiter=",")
            
            for testmode in self.testmode_list:
                print(f"{testmode} for {mapname} is done")

                if testmode == "Benchmark": 
                    self.Max_iter = 7           
                elif testmode == "perception_noise" or testmode == "Outputnoise_speed" or testmode == "Outputnoise_steering":
                    self.Max_iter = 30
                elif testmode == "control_delay_speed" or testmode == "control_Delay_steering" or testmode == "perception_delay":
                    self.Max_iter = 10

                for iter in range(self.Max_iter):
                    # self.trajectory_plot(mapname, testmode, iter+1)
                    self.speed_profile_plot(mapname, testmode, iter+1)
                    self.tracking_error(mapname, testmode, iter+1)


    def old_new_trajectory(self):
        for mapname in self.map_name_list:
        
            old_path = np.loadtxt(f"./old_maps/{mapname}_raceline.csv", delimiter=",")
            new_path = np.loadtxt(f"./maps/{mapname}_raceline.csv", delimiter= ",")

            plt.figure()
            plt.plot(old_path[:,1],old_path[:,2])
            plt.plot(new_path[:,1],new_path[:,2])
            plt.show()
            plt.clf()
                

    def trajectory_plot(self, mapname, testmode, iter):
        folder = "trajectory"
        # laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error, noise, completion
        data = np.loadtxt(f"./Imgs/{mapname}/{testmode}/{iter}.csv", delimiter=",")
        plt.title("Planned vs Car trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(data[:,1], data[:,2],label = "Car Trajectory")
        plt.plot(self.planned_path[:,1],self.planned_path[:,2], label = "Planned Trajectory")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./plots/{folder}/{mapname}/{testmode}/{iter}.svg")
        plt.clf()

    def speed_profile_plot(self, mapname, testmode, iter):
        folder = "speed_profile"
        # laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error, noise, completion
        data = np.loadtxt(f"./Imgs/{mapname}/{testmode}/{iter}.csv", delimiter=",")
        plt.title("Actual speed vs Expected speed")
        plt.xlabel("Time(s)")
        plt.ylabel("Velocity(m/s)")
        plt.plot(data[:,0], data[:,3],label = "Actual speed")
        plt.plot(data[:,0], data[:,4],label = "Expected speed")
        plt.legend()
        # plt.show()
        plt.grid(True)
        plt.savefig(f"./plots/{folder}/{mapname}/{testmode}/{iter}.svg")
        plt.clf()

    def tracking_error(self, mapname, testmode, iter):
        folder = "tracking_error"
        # laptime, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error, noise, completion
        data = np.loadtxt(f"./Imgs/{mapname}/{testmode}/{iter}.csv", delimiter=",")
        plt.title("Tracking error vs Time")
        plt.xlabel("Time(s)")
        plt.ylabel("Tracking error(m)")
        plt.plot(data[:,0], data[:,5],label = "Actual speed")
        plt.legend()
        # plt.show()
        plt.grid(True)
        plt.savefig(f"./plots/{folder}/{mapname}/{testmode}/{iter}.svg")
        plt.clf()

    def noise_success(self, mapname, testmode, iter, control_method):
        PurePursuitData = np.loadtxt(f"{control_method[0]}/csv/{mapname}/{mapname}_{testmode}.csv",delimiter=',',skiprows=1)
        MPCData = np.loadtxt(f"{control_method[1]}/csv/{mapname}/{mapname}_{testmode}.csv",delimiter=',',skiprows=1)
        
        PP_values = [float(row[2]) for row in PurePursuitData]
        PurePursuitData_2d = np.array(PP_values).reshape(-1,10)
        AveComp_PPData_2d = np.mean(PurePursuitData_2d,axis=1)

        PP_values = [float(row[5]) for row in PurePursuitData]
        PurePursuitData_2d = np.array(PP_values).reshape(-1,10)
        ATE_PPData_2d = np.mean(PurePursuitData_2d,axis=1)

        PP_values = [float(row[3]) for row in PurePursuitData]
        PurePursuitData_2d = np.array(PP_values).reshape(-1,10)
        AveNoiseScale_PPData_2d = np.mean(PurePursuitData_2d,axis=1)

        MPC_values = [float(row[2]) for row in MPCData]
        MPCData_2d = np.array(MPC_values).reshape(-1,10)
        AveComp_MPCData_2d = np.mean(MPCData_2d,axis=1)

        MPC_values = [float(row[5]) for row in MPCData]
        MPCData_2d = np.array(MPC_values).reshape(-1,10)
        ATE_MPCData_2d = np.mean(MPCData_2d,axis=1)

        MPC_values = [float(row[3]) for row in MPCData]
        MPCData_2d = np.array(MPC_values).reshape(-1,10)
        AveNoiseScale_MPCData_2d = np.mean(MPCData_2d,axis=1)

        ax1 = plt.subplot(211)
        plt.title("Average Track Progress vs " + testmode)
        plt.ylabel("Average Track Progress(%)")
        plt.plot(AveNoiseScale_PPData_2d,AveComp_PPData_2d,label = "Pure Pursuit")
        plt.plot(AveNoiseScale_MPCData_2d, AveComp_MPCData_2d, label = "MPC")
        plt.grid(True)
        plt.legend()
        # plt.savefig(f"results/{map_name}/{TESTMODE}.svg")

        ax2 = plt.subplot(212,sharex = ax1)
        plt.ylabel("Average tracking error(m)")
        plt.xlabel(testmode)
        plt.plot(AveNoiseScale_PPData_2d,ATE_PPData_2d,label = "Pure Pursuit")
        plt.plot(AveNoiseScale_MPCData_2d, ATE_MPCData_2d, label = "MPC")
        plt.grid(True)
        plt.savefig(f"results/{mapname}/{testmode}.svg")
        plt.close('all')


if __name__ == "__main__":
    plot = graphPlot()
    plot.plot_all()


