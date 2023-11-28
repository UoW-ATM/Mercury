import matplotlib.pyplot as plt
from math import floor
import pandas as pd

from . import unit_conversions as uc


class TrajectoryComponent:

	status_codes = ["0 - Computed ok",
					"1 - Not defined",
					"2 - Descent performances out of scope for weight/altitude",
					"3 - Climb performances out of scope for weight/altitude",
					"4 - Weight over MTOW during cruise",
					"5 - Climb during cruise out of scope for weight/altiude",
					"6 - FL too high for such a short fp distance",
					"7 - Climb at the very last before descendint, FL too high",
					"8 - Cruise segment starts with climb",
					"9 - Forced descent"]
	 
	def print_info(self):
		print("Trajectory component")


class TrajectorySegment(TrajectoryComponent):

	def __init__(self, fl_0, fl_1, distance, time, fuel, weight_0,
		weight_1, segment_type, avg_wind=None):
		self.fl_0 = fl_0
		self.fl_1 = fl_1
		self.distance = distance
		self.time = time
		self.fuel = fuel
		self.weight_0 = weight_0
		self.weight_1 = weight_1
		if time>0:
			self.avg_m = uc.kt2m((distance / (time / 60)), (fl_0+fl_1)/2)
		else:
			self.avg_m = 0
		self.segment_type = segment_type
		self.avg_wind = avg_wind
		self.status = 0
		
	def print_info(self):
		print("STATUS: "+TrajectorySegment.status_codes[self.status])
		# if self.status == 0:
		print("FL0: "+str(self.fl_0)+" FL1: "+str(self.fl_1)+" \n"+
			  "AVG_M: "+str(self.avg_m)+" \n"+
			  "DIST: "+str(self.distance)+
			  " TIME: "+str(floor(self.time/60))+":"+str(floor(self.time%60))+
		 ":"+ str(int(round(60*(self.time%1))))+" \n"+
			  "FUEL: "+str(self.fuel)+" WEIGHT 0: "+str(self.weight_0)+" WEIGHT 1: "+str(self.weight_1)+
			  " TYPE: "+str(self.segment_type)+" \n"+
			  " AVG WIND: "+str(self.avg_wind))


class Trajectory(TrajectoryComponent):

	def __init__(self, ac_icao, ac_model="", bada_version=-1, oew=-1, mpl=-1, distance_orig_fp=0):
		self.trajectory_segments = []
		self.distance = 0
		self.distance_orig_fp = distance_orig_fp
		self.time = 0
		self.fuel = 0
		self.weight_0 = 0
		self.weight_1 = 0
		self.fl_0 = 0
		self.fl_1 = 0
		self.fl_max = 0
		self.status = 1
		self.ac_icao = ac_icao
		self.ac_model = ac_model
		self.bada_version = bada_version
		self.oew = oew
		self.mpl = mpl

	def compress_segments(self):
		# Compress consecutive cruise segments into one
		compressed_trajectory_segments = []

		current_segment = self.trajectory_segments[0]

		i = 1
		saved = False

		while i < len(self.trajectory_segments):
			if (current_segment.segment_type == "cruise"
			   and self.trajectory_segments[i].segment_type == "cruise"
			   and self.trajectory_segments[i].fl_0 == current_segment.fl_0):
				# current segment is equal to i segment on flight levels

				current_segment.distance = current_segment.distance + self.trajectory_segments[i].distance
				current_segment.time = current_segment.time + self.trajectory_segments[i].time
				current_segment.fuel = current_segment.fuel + self.trajectory_segments[i].fuel
				current_segment.weight_1 = self.trajectory_segments[i].weight_1
				if (current_segment.avg_wind is not None) and (self.trajectory_segments[i].avg_wind is not None):
					current_segment.avg_wind = (((current_segment.avg_wind*current_segment.distance) + 
												(self.trajectory_segments[i].avg_wind*self.trajectory_segments[i].distance))/ 
												((current_segment.distance+self.trajectory_segments[i].distance)))

				if current_segment.time>0:
					current_segment.avg_m = uc.kt2m((current_segment.distance / (current_segment.time / 60)),
						(current_segment.fl_0+current_segment.fl_1)/2)
				else:
					current_segment.avg_m = 0

			else:
				saved = True
				compressed_trajectory_segments = compressed_trajectory_segments + [current_segment]   
				current_segment = self.trajectory_segments[i]
				saved = False

			i += 1

		if not saved:
			compressed_trajectory_segments = compressed_trajectory_segments + [current_segment]

		self.trajectory_segments = compressed_trajectory_segments

	def payload_carried(self):
		return self.weight_1-self.oew

	def number_of_climb_steps(self):
		ncs = 0
		for t in self.trajectory_segments:
			if t.segment_type == "climb":
				ncs = ncs + 1
		ncs = ncs - 1  # To remove the initial climb
		return ncs


	def get_trajectory_id(self):
		return self.ac_icao+"_"\
			   +str(int(round(self.distance_orig_fp)))+"_"\
			   +str(round(self.payload_carried()))+"_"\
			   +str(self.fl_max)+"_"\
			   +str(self.number_of_climb_steps())+"_"\
			   +str(self.bada_version)

	def dataframe_trajectory(self):
		trajectory_id=self.get_trajectory_id()
		y=[trajectory_id, 
		   self.ac_icao, self.ac_model, self.bada_version, int(round(self.distance, 0)),
		   int(round(self.distance_orig_fp,0)),
		   self.number_of_climb_steps(), 
		   float(round(self.time,3)),
		   float(round(self.fuel,3)), float(round(self.weight_0, 3)),
		   float(round(self.weight_1,3)), 
		   float(self.oew), float(self.mpl),
		   float(round((self.payload_carried()), 3)),
		   float(round((self.payload_carried())/self.mpl, 2)),
		   float(self.fl_0),
		   float(self.fl_1), 
		   float(self.fl_max),
		   self.status]
		d=pd.DataFrame(y).T
		d.columns=['trajectory_id', 'ac_icao', 'ac_model', 'bada_version', 'fp_distance',
		'fp_distance_orig', 
		'number_of_climb_steps',
		'fp_time', 'fp_fuel', 'fp_weight_0', 'fp_weight_1', 'oew', 'mpl', 'pl', 'pl_perc',
		'fp_fl_0', 'fp_fl_1', 'fp_fl_max', 'fp_status']
		return d

	def dataframe_trajectory_segments(self):
		trajectory_id = self.get_trajectory_id()

		x = []
		i = 0
		for ts in self.trajectory_segments:
			x = x + [[trajectory_id, i, ts.segment_type, int(round(ts.distance, 0)),
					  float(round(ts.time, 3)), float(round(ts.fuel, 3)),
					  float(round(ts.weight_0, 3)), float(round(ts.weight_1, 3)),
					  float(ts.fl_0), float(ts.fl_1), float(ts.avg_m), ts.status, ts.avg_wind]]
			i = i + 1

		d = pd.DataFrame(x)
		d.columns = ['trajectory_id', 'segment_order', 'segment_type', 'segment_distance', 'segment_time', 'segment_fuel',
		 'segment_weight_0', 'segment_weight_1', 'segment_fl_0', 'segment_fl_1', 'segment_avg_m', 'segment_status', 'avg_wind']
		return d

	def update_fl_weights_status(self):
		self.weight_0 = self.trajectory_segments[0].weight_0
		self.weight_1 = self.trajectory_segments[len(self.trajectory_segments)-1].weight_1
		self.fl_0 = self.trajectory_segments[0].fl_0
		self.fl_1 = self.trajectory_segments[len(self.trajectory_segments)-1].fl_1

	def add_back_trajectory_segment(self, ts):
		self.trajectory_segments = self.trajectory_segments + [ts]
		self.distance = self.distance + ts.distance
		self.time = self.time + ts.time
		self.fuel = self.fuel + ts.fuel
		self.fl_max = max(self.fl_max, max(ts.fl_0, ts.fl_1))
		self.update_fl_weights_status()

	def add_front_trajectory_segment(self, ts):
		self.trajectory_segments = [ts] + self.trajectory_segments
		self.distance = self.distance + ts.distance
		self.time = self.time + ts.time
		self.fuel = self.fuel + ts.fuel
		self.fl_max = max(self.fl_max, max(ts.fl_0, ts.fl_1))
		self.update_fl_weights_status()
		
	def add_back_trajectory(self, t):
		for ts in t.trajectory_segments:
			self.add_back_trajectory_segment(ts)
		self.update_fl_weights_status()

	def add_front_trajectory(self, t):
		for ts in reversed(t.trajectory_segments):
			self.add_front_trajectory_segment(ts)
		self.update_fl_weights_status()

	def compress_trajectory_segments(self):
		ts = self.trajectory_segments
		self.trajectory_segments = [ts[0]]

		i = 1
		while i < len(ts):
			if (ts[i].fl_0 == ts[i].fl_1) and (ts[i].fl_0 == ts[i-1].fl_0) and (ts[i].fl_1 == ts[i-1].fl_1):
				# The origin and end flight level are the same for both segment
				pass
			else:
				self.trajectory_segments = self.trajectory_segments + [ts[i]]

			i += 1

	def print_info(self, print_segments=True):
		print("STATUS: "+Trajectory.status_codes[self.status])
		print("AC: "+str(self.ac_icao)+" - "+str(self.ac_model))
		print("BADA: "+str(self.bada_version))
		print("DISTANCE ORIG FP: "+str(self.distance_orig_fp))
		print("DISTANCE: "+str(self.distance))
		print("FUEL: "+str(self.fuel))
		print("TIME: "+str(floor(self.time/60))+":"+str(floor(self.time%60)) +
			 ":"+ str(int(round(60*(self.time%1)))))
		print("WEIGHT_0: "+str(self.weight_0))
		print("WEIGHT_1: "+str(self.weight_1))
		if print_segments:
			print("*****")
			for ts in self.trajectory_segments:
				ts.print_info()
				print("--")   
			print("*****")
		
	def plot_distance_fl(self, show_plot=True):
		if len(self.trajectory_segments) > 0:
			fl = [self.trajectory_segments[0].fl_0]
			distance = [0]
			time = [0]
			fuel = [0]
			for i in (range(len(self.trajectory_segments))):
				fl = fl + [self.trajectory_segments[i].fl_1]
				distance = distance + [distance[len(distance)-1] + self.trajectory_segments[i].distance]
				time = time + [time[len(time)-1]+self.trajectory_segments[i].time]
				fuel = fuel + [fuel[len(fuel)-1]+self.trajectory_segments[i].fuel]

			plt.plot(distance, fl, label="FL"+str(int(self.fl_max)))

			plt.xlabel('distance (nm)')
			plt.ylabel('FL')
			if show_plot:
				plt.show()
