import numpy as np

# COUNTING METRICS
def computeMOE(GT, EST, fr):
	nt = GT.count_at_frame_OTS[fr] if fr in GT.count_at_frame_OTS else 0
	nt_est = EST.get_from_frame(fr)['inst_count']
	#return abs(nt_est-nt)/(max(nt,1))
	return abs(nt_est-nt)

def computeMRE(GT, EST, fr):
	pt = GT.count_at_frame[fr] if fr in GT.count_at_frame else 0
	nt_est = EST.get_from_frame(fr)['inst_count']
	#return abs(nt_est-pt)/max(pt,1)
	return abs(nt_est-pt)

def computeCOE(GT, EST):
	nT = GT.get_final_cumulative()
	nT_est = EST.get_final_cumulative()
	return abs(nT-nT_est)/max(nT,1)
	#return abs(nT-nT_est)

def computeCPE(GT, EST):
	pT = len(GT.IDS)
	nT_est = EST.get_final_cumulative()
	return abs(pT-nT_est)/max(pT,1)
	

def computeCCRE(GT, EST):
	nT = GT.get_final_cumulative()
	nT_est = EST.get_final_cumulative()
	return abs(nT-nT_est)/max(nT,1)
	#return abs(nT-nT_est)

def computeTCOE(GT, EST, fr, timerange):
	# Number of different ids present between the two time instances
	nT = GT.get_cumulativeOTS_between(fr-timerange, fr)
	nT_est = EST.get_cumulativeOTS_between(fr-timerange, fr)
	#print('{} / {} = {}'.format(nT, nT_est, abs(nT-nT_est)))
	return abs(nT-nT_est)

def computeMOE_distance(GT, EST, fr, mode):

	# Get distance (area) percentiles
	p50 = GT.get_percentiles_distance()

	# Get counts per each distance cluster (far, close) based on the percentiles
	gt_close, gt_far = GT.get_counts_per_distance(fr, p50, mode=mode)
	est_close, est_far  = EST.get_counts_per_distance(fr, p50)
	
	# Compute MOE
	MOE_close = abs(est_close-gt_close)
	MOE_far = abs(est_far-gt_far)
	return (MOE_close, MOE_far)


def computeCOEt(GT, EST, fr, timerange):
	# Number of different ids present between the two time instances
	nT = GT.get_cumulativeOTS_between(fr, fr+timerange-1)
	nT_est = EST.get_cumulativeOTS_between(fr, fr+timerange-1)
	return abs(nT-nT_est)

def computeCPEt(GT, EST, fr, timerange):
	# Number of different ids present between the two time instances
	nT = GT.get_cumulativeALL_between(fr, fr+timerange-1)
	nT_est = EST.get_cumulativeOTS_between(fr, fr+timerange-1)
	return abs(nT-nT_est)

def annotationPeopleBetween(GT, fr, timerange):
	# Number of different ids present between the two time instances
	return GT.get_cumulativeOTS_between(fr, fr+timerange-1)


# DETECTION METRICS
def computeTP_FP_FN(_GT, EST, fr, mode):

	# Get ANNO
	gt = []
	GT = _GT.get_from_frame(fr)
	for ID in GT:
		if mode in GT[ID]:
			gt.append(GT[ID][mode])
	gt = np.array(gt)

	# Get DET
	est = EST.get_from_frame(fr)['bboxes']

	TP, FP, FN = TP_FP_FN(gt, est, _GT.mask, ret='count')

	return TP, FP, FN


def computeTP_FP_FN_distance(GT, EST, fr, mode):
	# Get distance (area) percentiles
	p50 = GT.get_percentiles_distance()

	# Get counts per each distance cluster (far, close) based on the percentiles
	gt_close, gt_far = GT.get_counts_per_distance(fr, p50, ret='bboxes', mode=mode)
	#est_close, est_far  = EST.get_counts_per_distance(fr, p50, ret='bboxes')
	est = EST.get_from_frame(fr)['bboxes']

	# Close
	TPclose, FPclose, FNclose = TP_FP_FN(gt_close, est, GT.mask, ret='count')

	# Far
	TPfar, FPfar, FNfar = TP_FP_FN(gt_far, est, GT.mask, ret='count')
	
	return (TPclose, FPclose, FNclose), (TPfar, FPfar, FNfar)


def computeTP_FP_FN_occlusion(GT, EST, fr, mode):
	
	# Get counts per each occlusion cluster (not occluded [0], partially occluded [1], heavily occluded [2])
	gt0, gt1, gt2= GT.get_counts_per_occlusion(fr, mode)
	est = EST.get_from_frame(fr)['bboxes']

	# Not occluded
	TP0, FP0, FN0 = TP_FP_FN(gt0, est, GT.mask, ret='count')
	
	# Partially occluded
	TP1, FP1, FN1 = TP_FP_FN(gt1, est, GT.mask, ret='count')

	# Heavily occluded
	TP2, FP2, FN2 = TP_FP_FN(gt2, est, GT.mask, ret='count')

	return (TP0,FP0,FN0), (TP1,FP1,FN1), (TP2,FP2,FN2)



def iou(bboxes1, bboxes2, thr=1/3):
	#Return iou shape: len(labels) x len(outputs)
	x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
	x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
	xA = np.maximum(x11, np.transpose(x21))
	yA = np.maximum(y11, np.transpose(y21))
	xB = np.minimum(x12, np.transpose(x22))
	yB = np.minimum(y12, np.transpose(y22))
	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
	iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

	if thr=='max_per_row':
		return np.argmax(iou, axis=1)
	else:
		return iou>=thr

def TP_FP_FN(labels, outputs, mask, ret='bboxes'):
	
	if (not isinstance(labels,np.ndarray)) or (not isinstance(outputs,np.ndarray)):
		print('Labels and/or estimations are not np.array')
		assert 1==0

	if len(labels)==0 or len(outputs)==0:
		if len(outputs):
			#Discard FPs that are inside the ignore area
			outputs = discard_FP_in_ignore_area(outputs, mask)

			if ret=='bboxes':
				return np.empty(0), outputs, np.empty(0)
			if ret=='count':
				return 0, len(outputs), 0

		if len(labels):
			if ret=='bboxes':
				return np.empty(0), np.empty(0), labels
			if ret=='count':
				return 0, 0, len(labels)

		else:
			if ret=='bboxes':
				return np.empty(0), np.empty(0), np.empty(0)
			if ret=='count':
				return 0, 0, 0
	else:
		ious = iou(labels, outputs)
		TP = outputs[ious.sum(0)!=0]
		FP = outputs[ious.sum(0)==0]
		FP = discard_FP_in_ignore_area(FP, mask)	# Discard FP that are inside the ignore area
		FN = labels[ious.sum(1)==0]

		if ret=='bboxes':
			return TP, FP, FN
		if ret=='count':
			return len(TP), len(FP), len(FN)


def discard_FP_in_ignore_area(bboxes, mask):
	newbboxes=[]
	for bbox in bboxes:
		x0 = int(bbox[0])
		y0 = int(bbox[1])
		x1 = int(bbox[2])
		y1 = int(bbox[3])
		if (~mask[y0:y1,x0:x1]).any():
			newbboxes.append(bbox)
		#else:
		#	print('Discarding a box in the ignore area')

	return np.array(newbboxes)

# AGE METRICS
def age_computeTP_FP_FN(_GT, EST, fr, mode, margin=2, vis=False):

	# Margin indicates the overlap between age classes

	# Get ANNO
	gt_bboxes = []
	gt_age = []
	GT = _GT.get_from_frame(fr)
	for ID in GT:
		if mode in GT[ID] and 'age' in GT[ID]:
			gt_bboxes.append(GT[ID][mode])
			gt_age.append(GT[ID]['age'])
	gt_bboxes = np.array(gt_bboxes)
	gt_age = np.array(gt_age)

	# Get DET
	est_bboxes = EST.get_from_frame(fr)['bboxes']
	est_age = EST.get_from_frame(fr)['age']

	if len(gt_bboxes) and len(est_bboxes):
		ious = iou(gt_bboxes, est_bboxes)
		# Matches
		# Rows: labels
		# Cols: output
		TPs_ind = np.transpose(ious.nonzero())
	else:
		TPs_ind = np.empty(0)

	if vis:
		if len(TPs_ind):
			true = {'bboxes': gt_bboxes[TPs_ind[:,0]], 'age': gt_age[TPs_ind[:,0]]}
			est = {'bboxes': est_bboxes[TPs_ind[:,1]], 'age': est_age[TPs_ind[:,1]]}
		else:
			true = None
			est = None
		return {'true': true, 'est': est}

	labels = range(0,4)
	TP = dict.fromkeys(labels)
	FP = dict.fromkeys(labels)
	FN = dict.fromkeys(labels)
	for label in labels:
		TP[label]=0
		FP[label]=0
		FN[label]=0
	
	#For each TP, check the gender agreement
	for ind in TPs_ind:
		gt = gt_age[ind[0]]
		est = est_age[ind[1],0]
		
		if gt == 0:	# Class 0 '<18'
			if est <= 18+margin:
				TP[0] += 1
			else:
				FN[0] += 1
				FP[age2class(est)] += 1
		
		elif gt == 1:	# Class 1 '18-34'
			if est >= 18-margin and est <= 34+margin:
				TP[1] += 1
			else:
				FN[1] += 1
				FP[age2class(est)] += 1

		elif gt == 2:	# Class 2 '35-65'
			if est >= 34-margin and est <= 65+margin:
				TP[2] += 1
			else:
				FN[2] += 1
				FP[age2class(est)] += 1

		elif gt_age[ind[0]] == 3:	# Class 3 '>65'
			if est >= 65-margin:
				TP[3] += 1
			else:
				FN[3] += 1
				FP[age2class(est)] += 1
			
	return TP, FP, FN

def ageclass_computeTP_FP_FN(_GT, EST, fr, mode, vis=False):

	# Margin indicates the overlap between age classes

	# Get ANNO
	gt_bboxes = []
	gt_age = []
	GT = _GT.get_from_frame(fr)
	for ID in GT:
		if mode in GT[ID] and 'age' in GT[ID]:
			gt_bboxes.append(GT[ID][mode])
			gt_age.append(GT[ID]['age'])
	gt_bboxes = np.array(gt_bboxes)
	gt_age = np.array(gt_age)

	# Get DET
	est_bboxes = EST.get_from_frame(fr)['bboxes']
	est_ageclass = EST.get_from_frame(fr)['age']

	if len(gt_bboxes) and len(est_bboxes):
		ious = iou(gt_bboxes, est_bboxes)
		# Matches
		# Rows: labels
		# Cols: output
		TPs_ind = np.transpose(ious.nonzero())
	else:
		TPs_ind = np.empty(0)

	if vis:
		if len(TPs_ind):
			true = {'bboxes': gt_bboxes[TPs_ind[:,0]], 'age': gt_age[TPs_ind[:,0]]}
			est = {'bboxes': est_bboxes[TPs_ind[:,1]], 'age': est_ageclass[TPs_ind[:,1]]}
		else:
			true = None
			est = None
		return {'true': true, 'est': est}


	labels = range(0,4)
	TP = dict.fromkeys(labels)
	FP = dict.fromkeys(labels)
	FN = dict.fromkeys(labels)
	for label in labels:
		TP[label]=0
		FP[label]=0
		FN[label]=0
	
	#For each TP, check the gender agreement
	for ind in TPs_ind:
		gt = gt_age[ind[0]]
		est = est_ageclass[ind[1],0]


		if gt == 0:	# Class 0 '<18'
			if est == 0:
				TP[0] += 1
			else:
				FN[0] += 1
				FP[est] += 1
		
		elif gt == 1:	# Class 1 '18-34'
			if est == 1:
				TP[1] += 1
			else:
				FN[1] += 1
				FP[est] += 1

		elif gt == 2:	# Class 2 '35-65'
			if est == 2:
				TP[2] += 1
			else:
				FN[2] += 1
				FP[est] += 1

		elif gt == 3:	# Class 3 '>65'
			if est == 3:
				TP[3] += 1
			else:
				FN[3] += 1
				FP[est] += 1
			
	return TP, FP, FN




def age2class(age):
	if age <= 18:
		ageclass=0
	elif age <= 34:
		ageclass=1
	elif age <= 65:
		ageclass=2
	else:
		ageclass=3
	return ageclass

# GENDER METRICS
def gender_computeTP_FP_FN(_GT, EST, fr, mode):

	# Get ANNO
	gt_bboxes = []
	gt_gender = []
	GT = _GT.get_from_frame(fr)
	for ID in GT:
		if mode in GT[ID] and 'gender' in GT[ID]:
			gt_bboxes.append(GT[ID][mode])
			gt_gender.append(GT[ID]['gender'])
	gt_bboxes = np.array(gt_bboxes)
	gt_gender = np.array(gt_gender)

	# Get DET
	est_bboxes = EST.get_from_frame(fr)['bboxes']
	est_gender = EST.get_from_frame(fr)['gender']

	if len(gt_bboxes) and len(est_bboxes):
		ious = iou(gt_bboxes, est_bboxes)
		# Matches
		# Rows: labels
		# Cols: output
		TPs_ind = np.transpose(ious.nonzero())
	else:
		TPs_ind = np.empty(0)

	labels = ['male','female']
	TP = dict.fromkeys(labels)
	FP = dict.fromkeys(labels)
	FN = dict.fromkeys(labels)
	for label in labels:
		TP[label]=0
		FP[label]=0
		FN[label]=0
	
	#For each TP, check the gender agreement
	#Order of the clases: male / female
	for ind in TPs_ind:
		
		# MALE
		#TP
		if gt_gender[ind[0]] == 'male':
			if est_gender[ind[1]] == 0:
				TP['male'] += 1
			elif est_gender[ind[1]] == 1:
				FN['male'] += 1
		#FP
		elif gt_gender[ind[0]] == 'female' and \
			est_gender[ind[1]] == 0 :
			FP['male'] += 1
			

		##############
		# FEMALE
		#TP
		if gt_gender[ind[0]] == 'female':
			if est_gender[ind[1]] == 1:
				TP['female'] += 1
			elif est_gender[ind[1]] == 0:
				FN['female'] += 1
		#FP
		elif gt_gender[ind[0]] == 'male' and \
			est_gender[ind[1]] == 1 :
			FP['female'] += 1


	return TP, FP, FN
