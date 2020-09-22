from xml.dom import minidom
import numpy as np

import re
import cv2

def isSamePerson(_ID,_id):
	ID = _ID if isinstance(_ID, int) else int(_ID.split('+')[0])
	id = _id if isinstance(_id, int) else int(_id.split('+')[0])
	return ID==id

def age2class(age, attribute_name):
	if attribute_name=='real_age':
		age = int(age)
		if age <= 18:
			ageclass=0
		elif age <= 34:
			ageclass=1
		elif age <= 65:
			ageclass=2
		else:
			ageclass=3

	elif attribute_name=='estimate_age':
		if age=='<18':
			ageclass=0
		elif age=='19-34':
			ageclass=1
		elif age=='35-65':
			ageclass=2
		elif age=='>65':
			ageclass=3

	return ageclass

class annotationReader:
	def __init__(self, anotationFileName, mode='person'):
		self.mode = mode
		self.data = {}
		self.IDS=set()
		self.IDS_OTS=set()
		self.IDS_lastSeen={}
		self.ID_actualID={}
		self.areas=[]		#to get person-to-signage distance
		self.final_cumulative = 0
		self.readMask(anotationFileName)
		self.readAnnotations(anotationFileName)
		self.first_frame = min(list(self.data.keys()))
		self.last_frame = max(list(self.data.keys()))
		self.calculate_count_at_frame()
		self.calculate_cumulative()
		self.calculate_counts_until_frame()
		self.max_count = max(max(self.count_at_frame.values()), max(self.count_at_frame_OTS.values()))

	def readMask(self, anotationFileName):
		self.mask = cv2.imread(anotationFileName.replace('.xml','.jpg'))[:,:,0].astype(bool)

	def check_if_ignore(self, x0, y0, x1, y1):
		x0 = int(x0)
		y0 = int(y0)
		x1 = int(x1)
		y1 = int(y1)
		return (self.mask[y0:y1,x0:x1]).all()

	def addNewKey(self, frame, ID):
		if frame not in self.data:
			self.data[frame] = {}
			#self.data[frame][ID] = dict.fromkeys(['person','occlussion','pose','orientation','attention','age','gender','area'])
		#else:
			#self.data[frame][ID] = dict.fromkeys(['person','occlussion','pose','orientation','attention','age','gender','area'])
		if ID not in self.data[frame]:
			self.data[frame][ID] = {}
	
	def get_annotation_resolution(self, xmldoc):
		metas = xmldoc.getElementsByTagName('meta')
		for meta in metas:
			tasks = meta.getElementsByTagName('task')
			for task in tasks:
				size = task.getElementsByTagName('original_size')
				self.original_frame_width 	= int(size[0].getElementsByTagName('width')[0].childNodes[0].nodeValue)
				self.original_frame_height 	= int(size[0].getElementsByTagName('height')[0].childNodes[0].nodeValue)
				return
	
	def readAnnotations(self, anotationFileName):
		xmldoc = minidom.parse(anotationFileName)
		
		# Get annotation metadata (resolution and framerate)
		self.get_annotation_resolution(xmldoc)
		
		# Video resolution
		self.annotated_res = '1080' if self.original_frame_height==1080 else '4k'

		# New ID is considered when an ID does not appear for 10seconds and then re-appears
		self.num_frames_newID = 10*30 if self.annotated_res=='1080' else 10*60 

		tracks = xmldoc.getElementsByTagName('track')
		ID_groupid = {}

		count=-1
		for track in tracks:
			if track.attributes['label'].value == 'person':

				boxes = track.getElementsByTagName('box')
				try:
					group_id = int(track.attributes['group_id'].value)
				except: 
					group_id = count
					count -= 1

				for box in boxes:	
					frame = int(box.attributes['frame'].value)

					scale = 1 if self.annotated_res=='1080' else 2
					x0 = float(box.attributes['xtl'].value)/scale
					y0 = float(box.attributes['ytl'].value)/scale
					x1 = float(box.attributes['xbr'].value)/scale
					y1 = float(box.attributes['ybr'].value)/scale

					#if self.check_if_ignore(x0, y0, x1, y1):
					#	continue

					if self.annotated_res=='4k':
						if frame%2:
							continue	# For 4k annotations do not use odd frames
						else:
							frame=int(frame/2)

					all_attributes = box.getElementsByTagName('attribute')
					the_attributes=[]
					for attribute in all_attributes:

						att_name = attribute.attributes['name'].value
						try:
							att_value = attribute.childNodes[0].nodeValue
						except:
							att_value = -1
							#print('attribute without value')

						if att_name=='ID':
							if not isinstance(att_value, int):
								ID = int(''.join(i for i in att_value if i.isdigit()))
							else:
								ID = att_value
							
							if ID in self.IDS_lastSeen:
								if frame-self.IDS_lastSeen[ID] >= self.num_frames_newID: 
									self.ID_actualID[ID] = '{}+'.format(self.ID_actualID[ID])
								self.IDS_lastSeen[ID] = frame
								ID = self.ID_actualID[ID]
							else:
								self.IDS_lastSeen[ID] = frame
								self.ID_actualID[ID] = ID

							ID_groupid[group_id] = ID

						elif (att_name=='real_age' or att_name=='estimate_age') and att_value != 'n/a':
							the_attributes.append(('age', age2class(att_value, att_name)))
						else:
							the_attributes.append((att_name, att_value))
						
					# Initialise dictionary if needed (for new frame or new ID on that frame)
					if (frame not in self.data) or (ID not in self.data[frame]):
						self.addNewKey(frame, ID)

					# Add annotation to dictionary of annotations

					self.data[frame][ID]['person'] = (x0,y0,x1,y1)
					for att_name, att_value in the_attributes:
						self.data[frame][ID][att_name]=att_value

						if att_name=='orientation' and att_value!='heading_opposite':
							self.IDS_OTS.add(ID)

					self.IDS.add(ID)

					if self.mode=='person':
						self.data[frame][ID]['area'] = (x1-x0)*(y1-y0)
						self.areas.append(self.data[frame][ID]['area'])

		count=-1
		for track in tracks:
			if track.attributes['label'].value == 'face':

				boxes = track.getElementsByTagName('box')
				try:
					ID = ID_groupid[int(track.attributes['group_id'].value)]
				except: 
					ID = count
					count -= 1

				for box in boxes:	
					frame = int(box.attributes['frame'].value)

					if self.annotated_res=='4k':
						if frame%2:
							continue	# For 4k annotations do not use odd frames
						else:
							frame=int(frame/2)

					if frame in self.data:
						for id in self.data[frame]:	
							if isSamePerson(ID, id):
								ID = id
					else:
						ID=id

					# Initialise dictionary if needed (for new frame or new ID on that frame)
					if (frame not in self.data) or (ID not in self.data[frame]):
						self.addNewKey(frame, ID)

					all_attributes = box.getElementsByTagName('attribute')
					the_attributes=[]
					for attribute in all_attributes:
						att_name = attribute.attributes['name'].value
						att_value = attribute.childNodes[0].nodeValue
						the_attributes.append((att_name, att_value))

					# Add annotation to dictionary of annotations
					scale = 1 if self.annotated_res=='1080' else 2
					x0 = float(box.attributes['xtl'].value)/scale
					y0 = float(box.attributes['ytl'].value)/scale
					x1 = float(box.attributes['xbr'].value)/scale
					y1 = float(box.attributes['ybr'].value)/scale

					self.data[frame][ID]['face'] = (x0,y0,x1,y1)
					
					for att_name, att_value in the_attributes:
						self.data[frame][ID][att_name]=att_value

					self.IDS.add(ID)
					
					if self.mode=='face':
						self.data[frame][ID]['area'] = (x1-x0)*(y1-y0)
						self.areas.append(self.data[frame][ID]['area'])


	def calculate_count_at_frame(self):
		#Only people with OTS
		self.count_at_frame_OTS = {}
		self.count_at_frame = {}
		self.person_annotations = []
		self.face_annotations= []
		for frame in self.data:
			persons=0
			faces=0
			self.count_at_frame[frame] = len(self.data[frame].keys())
			self.count_at_frame_OTS[frame] = 0
			for ID in self.data[frame]:

				if 'person' in self.data[frame][ID]:
					persons += 1

				if 'face' in self.data[frame][ID]:
					faces += 1

				if 'person' in self.data[frame][ID] and self.data[frame][ID]['orientation'] != 'heading_opposite':
					self.count_at_frame_OTS[frame] += 1

			self.person_annotations.append(persons)
			self.face_annotations.append(faces)

			
	
	def calculate_counts_until_frame(self):
		self.count_until_fr = []
		self.count_until_fr_OTS = []
		for fr in range(0,self.last_frame+1):
			tmp1=[]
			tmp2=[]
			for i in range(0,fr+1):
				tmp1.append(self.count_at_frame[i] if i in self.count_at_frame else 0)
				tmp2.append(self.count_at_frame_OTS[i] if i in self.count_at_frame_OTS else 0)

			self.count_until_fr.append(tmp1)
			self.count_until_fr_OTS.append(tmp2)
	

	def calculate_cumulative(self):
		
		# Frame | [ids]
		self.idsOTS_per_frame = []
		self.idsALL_per_frame = []
		for frame in range(self.last_frame):
			idsALL_currentframe=[]
			idsOTS_currentframe=[]
			if frame in self.data:
				for ID in self.data[frame]:
					idsALL_currentframe.append(ID)
					if 'person' in self.data[frame][ID] and self.data[frame][ID]['orientation'] != 'heading_opposite':
						idsOTS_currentframe.append(ID)
			self.idsALL_per_frame.append(idsALL_currentframe)
			self.idsOTS_per_frame.append(idsOTS_currentframe)


	def get_cumulativeOTS_between(self, fr0, fr1):
		ids=set()
		for fr in range(fr0,fr1):
			if fr < len(self.idsOTS_per_frame):
				for id in self.idsOTS_per_frame[fr]:
					ids.add(id)
		return len(ids)	

	def get_cumulativeALL_between(self, fr0, fr1):
		ids=set()
		for fr in range(fr0,fr1):
			if fr < len(self.idsOTS_per_frame):
				for id in self.idsOTS_per_frame[fr]:
					ids.add(id)
		return len(ids)	

	def get_from_frame(self, frame):
		if frame in self.data:
			return self.data[frame]
		else:
			return np.empty(0)


	def get_final_cumulative(self):
		return len(self.IDS_OTS)

	def get_percentiles_distance(self):
		#p30 = np.percentile(self.areas, 30)	#Far (small)
		#p70 = np.percentile(self.areas, 70)	#Close (large)
		#return p30, p70
 
		return np.percentile(self.areas, 50)

	def get_percentiles_density(self):
		all_counts = list(self.count_at_frame.values())
		#self.p30_density = np.percentile(all_counts, 30) # Sparse (less)
		#self.p70_density = np.percentile(all_counts, 70) # Dense (more)

		self.p50_density = np.percentile(all_counts, 50)

	def get_counts_per_distance(self, frame, p50, ret='count', mode='face'):
		if ret=='count':
			far = 0
			close = 0
			if frame in self.data:
				for ID in self.data[frame]:
					try:
						if self.data[frame][ID]['area'] < p50:
							far += 1
						else: 
							close += 1
					except:
						continue
		if ret=='bboxes':
			far = []
			close = []
			if frame in self.data:
				for ID in self.data[frame]:
					if mode in self.data[frame][ID]:
						if self.data[frame][ID]['area'] < p50:
							far.append(self.data[frame][ID][mode])
						else: 
							close.append(self.data[frame][ID][mode])
			far = np.array(far)
			close = np.array(close)

		return close, far

	def get_counts_per_occlusion(self, frame, mode):
		not_occ  = []
		part_occ = []
		heav_occ = []
		if frame in self.data:
			for ID in self.data[frame]:
				if mode in self.data[frame][ID]:
					if self.data[frame][ID]['occlusion'] == 'not_occluded':
						not_occ.append(self.data[frame][ID][mode])
					elif self.data[frame][ID]['occlusion'] == 'partially_occluded':
						part_occ.append(self.data[frame][ID][mode])
					elif self.data[frame][ID]['occlusion'] == 'fully_occluded':
						heav_occ.append(self.data[frame][ID][mode])
		return np.array(not_occ), np.array(part_occ), np.array(heav_occ)


class estimationReader:
	def __init__(self, estimationFileName, bodypart=None, mask=None):
		self.bodypart = bodypart
		self.mask = mask
		self.data = {}
		self.readEstimations(estimationFileName)
		self.first_frame = min(list(self.data.keys()))
		self.last_frame = max(list(self.data.keys()))
		self.calculate_counts_until_frame()

	def addNewKey(self, frame):
		if frame not in self.data:
			self.data[frame] = dict.fromkeys(['bboxes','inst_count','cum_count','area','time','ids','age','gender'])
	
	def check_if_ignore(self, bboxes, areas):
		keep = []
		for i, box in enumerate(bboxes):
			if (~self.mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])]).any():
				keep.append(i)
		
		bboxes = bboxes[keep]
		areas = areas[keep]

	def readEstimations(self, estimationFileName):

		f = open(estimationFileName, 'r')
		count=0
		all_ids = set()
		for frame, line in enumerate(f):
			parts = line.split(',')
			#frame = int(parts[0])
			#inst_count = int(parts[1])
			#cum_count = int(parts[2])
			time = float(parts[0])
			
			# Localization
			if self.bodypart=='person':
				bboxes = np.empty(0)
				areas = np.empty(0)
				x0 = np.array(parts[1::11]).astype(float).reshape(-1,1)
				y0 = np.array(parts[2::11]).astype(float).reshape(-1,1)
				x1 = np.array(parts[3::11]).astype(float).reshape(-1,1)
				y1 = np.array(parts[4::11]).astype(float).reshape(-1,1)
				

			elif self.bodypart=='face':
				bboxes = np.empty(0)
				areas = np.empty(0)
				x0 = np.array(parts[5::11]).astype(float).reshape(-1,1)
				y0 = np.array(parts[6::11]).astype(float).reshape(-1,1)
				x1 = np.array(parts[7::11]).astype(float).reshape(-1,1)
				y1 = np.array(parts[8::11]).astype(float).reshape(-1,1)
			
			bboxes = np.hstack((x0,y0,x1,y1))
			areas = (x1-x0)*(y1-y0)
			
			ids 		= np.array(parts[9::11]).astype(int).tolist()
			all_ids.update(ids)
			age 		= np.array(parts[10::11]).astype(int).reshape(-1,1)
			gender 		= np.array(parts[11::11]).astype(int).reshape(-1,1)			

			# Initialise dictionary if needed (for new frame or new ID on that frame)
			if frame not in self.data:
				self.addNewKey(frame)

			self.data[frame]['bboxes'] = bboxes
			self.data[frame]['inst_count'] = len(bboxes)
			self.data[frame]['cum_count'] = len(all_ids)
			self.data[frame]['ids'] = ids
			self.data[frame]['area'] = areas
			self.data[frame]['time'] = time
			
			self.data[frame]['age'] = age
			self.data[frame]['gender'] = gender
			
				

	def get_from_frame(self, frame):
		if frame in self.data:
			return self.data[frame]
		else:
			return np.empty(0)

	
	def calculate_counts_until_frame(self):
		self.count_until_fr = []
		self.max_count=0
		for fr in range(0,self.last_frame+1):
			tmp=[]
			for i in range(0,fr+1):
				tmp.append(self.data[i]['inst_count'] if i in self.data else 0)
			self.count_until_fr.append(tmp)
			self.max_count = max(self.max_count, max(tmp))
	

	def get_final_cumulative(self):
		return self.data[self.last_frame]['cum_count']

	def get_cumulativeOTS_between(self, fr0, fr1):
		ids=set()
		for fr in range(fr0,fr1):
			try:
				for id in self.data[fr]['ids']:
					ids.add(id)
			except:
				continue

		return len(ids)

	def get_counts_per_distance(self, frame, p50, ret='count'):
		if ret=='count':
			far=0
			close=0
			if len(self.data[frame]['bboxes']):
				far = (self.data[frame]['area'] < p50).sum()
				close = (self.data[frame]['area'] >= p50).sum()
		
		if ret=='bboxes':
			far=np.empty(0)
			close=np.empty(0)
			if len(self.data[frame]['bboxes']):
				far = self.data[frame]['bboxes'][(self.data[frame]['area']<p50).reshape(-1)]
				close = self.data[frame]['bboxes'][(self.data[frame]['area']>=p50).reshape(-1)]

		return close, far