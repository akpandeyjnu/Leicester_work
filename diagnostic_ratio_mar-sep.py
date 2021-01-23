# This code compare GEOS-Chem model and DEFRA sites ammonia 
# Please contact Alok Pandey ap744@leicester.ac.uk for any further clarifications or details

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler
import datetime
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.cm as cm
import glob
from scipy import stats
#from bootstrap import rma
from scipy.stats import gaussian_kde

#Use todays date to plot files - good practice to save files name with date
Today_date=datetime.datetime.now().strftime("%Y%m%d")

###Different cmap options
# cmap = matplotlib.cm.get_cmap('brewer_RdBu_11')
# cmap = cm.jet
cmap = cm.rainbow
#cmap = cm.YlOrRd

def NH3():
	#read UKEAP ammonia datasets here scratch_alok -> /scratch/uptrop/ap744
	UKEAPpath='/scratch/uptrop/ap744/UKEAP_data/UKEAP_NH3_particulate_ammonium/gaseousAmmonia_Active/'
	ammonia_files=glob.glob(UKEAPpath + '27-UKA0*-2016_active_*.csv')
	#print (ammonia_files)

	# read csv file having DEFRA sites details
	sites = pd.read_csv('/scratch/uptrop/ap744/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_NH3_sites_details.csv', encoding= 'unicode_escape')
	#print (sites.head(10))
	ID = sites["UK-AIR_ID"]
	#print (ID)

	# site wise annual mean computation  
	x = []
	for f in ammonia_files:
		df = pd.read_csv(f,parse_dates=["Start Date", "End Date"])  
		#print (df.head(5))
		#print (len(ammonia_files))
		sitesA = sites.copy()
		#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

		#Annual Mean calculation
		mean_A= df["Measurement"].mean() # to compute annual mean
		#print (mean_A, f[89:97])
		
		#MAMJJAS mean Calculation
		msp_start = pd.to_datetime("15/02/2016")
		msp_end = pd.to_datetime("15/10/2016")
		msp_subset = df[(df["Start Date"] > msp_start) & (df["End Date"] < msp_end)]
		mean_msp = msp_subset["Measurement"].mean()
		
		#MAM mean Calculation
		mam_start = pd.to_datetime("15/02/2016")
		mam_end = pd.to_datetime("15/06/2016")
		mam_subset = df[(df["Start Date"] > mam_start) & (df["End Date"] < mam_end)]
		mean_mam = mam_subset["Measurement"].mean()
		
		#JJA mean Calculation
		jja_start = pd.to_datetime("15/05/2016")
		jja_end = pd.to_datetime("15/09/2016")
		jja_subset = df[(df["Start Date"] > jja_start) & (df["End Date"] < jja_end)]
		mean_jja = jja_subset["Measurement"].mean()

		#SON mean Calculation
		son_start = pd.to_datetime("15/08/2016")
		son_end = pd.to_datetime("15/11/2016")
		son_subset = df[(df["Start Date"] > son_start) & (df["End Date"] < son_end)]
		mean_son = son_subset["Measurement"].mean()
		
		#DJF mean Calculation
		
		d_start = pd.to_datetime("15/11/2016")
		d_end = pd.to_datetime("31/12/2016")
		d_subset = df[(df["Start Date"] > d_start) & (df["End Date"] < d_end)]
		mean_d = d_subset["Measurement"].mean()
		#print (mean_d, 'mean_d')
		
		
		jf_start = pd.to_datetime("01/01/2016")
		jf_end = pd.to_datetime("15/03/2016")
		jf_subset = df[(df["Start Date"] > jf_start) & (df["End Date"] < jf_end)]
		mean_jf = jf_subset["Measurement"].mean()
		#print (mean_jf, 'mean_jf')
		
		
		mean_djf_a  = np.array([mean_d, mean_jf])
		
		mean_djf = np.nanmean(mean_djf_a, axis=0)
		#print (mean_djf, 'mean_djf')
		
		sitesA["ammonia_annual_mean"] = mean_A
		sitesA["ammonia_msp_mean"] = mean_msp
		sitesA["ammonia_mam_mean"] = mean_mam
		sitesA["ammonia_jja_mean"] = mean_jja
		sitesA["ammonia_son_mean"] = mean_son
		sitesA["ammonia_djf_mean"] = mean_djf
		#print (sitesA.head(50))
		
		x.append(
		{
			'UK-AIR_ID':f[89:97],
			'ammonia_annual_mean':mean_A,
			'ammonia_msp_mean':mean_msp,
			'ammonia_mam_mean':mean_mam,
			'ammonia_jja_mean':mean_jja,
			'ammonia_son_mean':mean_son,
			'ammonia_djf_mean':mean_djf
			}
			)
		#print (x)
		
	id_mean = pd.DataFrame(x)
	#print (id_mean.head(3))

	df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
	#print (df_merge_col.head(50))

	#####export csv file having site wise annual mean information if needed 
	#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/ammonia_annual_mean.csv')

	#drop extra information from pandas dataframe
	df_merge_colA = df_merge_col.drop(['S No'], axis=1)
	df_merge_colA = df_merge_colA.drop(df_merge_colA.index[[0,3,4,6,7,9,11,13,14,16,22,25,28,29,30,33,34,35,37,40,40,41,42,46,47]])
	df_merge_colA.reset_index(drop=True, inplace=True)
	#print (df_merge_colA.head(55))

	df_merge_colB = df_merge_colA.copy()

	###################################################################################
	###########  Delete Data over Scotland           ##################################
	###################################################################################
	df_merge_colB.drop(df_merge_colB[df_merge_colB['Lat'] > 56].index, inplace = True) 
	#print(df_merge_colB.head(11)) 
	df_merge_colB.reset_index(drop=True, inplace=True)
	#print(df_merge_colB.head(11)) 


	# change datatype to float to remove any further problems
	df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
	df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)
	df_merge_colB['Long'] = df_merge_colB['Long'].astype(float)
	df_merge_colB['Lat'] = df_merge_colB['Lat'].astype(float)

	#get sites information
	sites_lon = df_merge_colA['Long']
	sites_lat = df_merge_colA['Lat']


	#get sites information for calculation
	sites_lon_c = df_merge_colB['Long']
	sites_lat_c = df_merge_colB['Lat']


	#getting annual mean data
	sites_ammonia_AM = df_merge_colA['ammonia_annual_mean']
	#seasonal mean data
	sites_ammonia_msp = df_merge_colA['ammonia_msp_mean']
	sites_ammonia_mam = df_merge_colA['ammonia_mam_mean']
	sites_ammonia_jja = df_merge_colA['ammonia_jja_mean']
	sites_ammonia_son = df_merge_colA['ammonia_son_mean']
	sites_ammonia_djf = df_merge_colA['ammonia_djf_mean']
	sites_name = df_merge_colA['Site_Name']
	#print (sites_ammonia_AM, sites_name, sites_lat, sites_lon)


	#seasonal mean data for calculation
	sites_ammonia_AM_c = df_merge_colB['ammonia_annual_mean']
	sites_ammonia_msp_c = df_merge_colB['ammonia_msp_mean']
	sites_ammonia_mam_c = df_merge_colB['ammonia_mam_mean']
	sites_ammonia_jja_c = df_merge_colB['ammonia_jja_mean']
	sites_ammonia_son_c = df_merge_colB['ammonia_son_mean']
	sites_ammonia_djf_c = df_merge_colB['ammonia_djf_mean']
	sites_name_c = df_merge_colB['Site_Name']



	#########################       Reading GEOS-Chem files    ################################
	Species = sorted(glob.glob("/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/SpeciesConc/2016/GEOSChem.SpeciesConc*.nc4"))  # iccw
	#print (Species)
	########################### 50% increase in NH3 Emission ##################################
	#Species = sorted(glob.glob("/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_scale_nh3_emis/SpeciesConc/2016/GEOSChem.SpeciesConc*.nc4"))  #scale Nh3 by 50%
	StateMet = sorted(glob.glob("/scratch/uptrop/ap744/GEOS-Chem_outputs/GEOSChem.StateMet.2016*b.nc4"))

	Species = Species[:] 
	StateMet = StateMet[:]
	#print(Species, StateMet, sep = "\n")

	Species  = [xr.open_dataset(file) for file in Species]
	StateMet = [xr.open_dataset(file) for file in StateMet]

	#ammonia sufrace layer
	GC_surface_ammonia = [data['SpeciesConc_NH3'].isel(time=0,lev=0) for data in Species]
	#print (GC_surface_ammonia)

	#Avogadro's number [mol-1]
	AVOGADRO = 6.022140857e+23

	# Typical molar mass of air [kg mol-1]
	MW_AIR = 28.9644e-3
	# convert unit for ammonia (dry mol/mol to ug/m3)
	surface_AIRDEN = [data['Met_AIRDEN'].isel(time=0,lev=0) for data in StateMet] #kg/m3
	surface_AIRNUMDEN_a = np.asarray(surface_AIRDEN)/MW_AIR #mol/m3
	surface_AIRNUMDEN_b = surface_AIRNUMDEN_a*AVOGADRO # unit molec air/m3
	surface_AIRNUMDEN = surface_AIRNUMDEN_b/1e6 #unit molec air/cm3
	surface_ammonia_mass  = [x*y*17/(6.022*1e11) for (x,y) in zip(GC_surface_ammonia,surface_AIRNUMDEN)]
	#print (surface_ammonia_mass)

	#Geos-Chem Annual Mean
	GC_surface_ammonia_AM = sum(surface_ammonia_mass)/len(surface_ammonia_mass)
	#print (GC_surface_ammonia_AM,'AnnualMean')
	#print (GC_surface_ammonia_AM.shape,'AnnualMean shape')
	#Geos-Chem seasonal Mean
	GC_surface_ammonia_msp = sum(surface_ammonia_mass[2:9])/len(surface_ammonia_mass[2:9])

	GC_surface_ammonia_mam = sum(surface_ammonia_mass[2:5])/len(surface_ammonia_mass[2:5])
	#print (GC_surface_ammonia_mam.shape, 'MAM-shape')
	GC_surface_ammonia_jja = sum(surface_ammonia_mass[5:8])/len(surface_ammonia_mass[5:8])
	#print (GC_surface_ammonia_jja)
	GC_surface_ammonia_son = sum(surface_ammonia_mass[8:11])/len(surface_ammonia_mass[8:11])
	#print (GC_surface_ammonia_son)
	GC_surface_ammonia_jf = sum(surface_ammonia_mass[0:2])/len(surface_ammonia_mass[0:2])
	#print (GC_surface_ammonia_jf, 'jf_shape')
	GC_surface_ammonia_d = surface_ammonia_mass[11]
	#print (GC_surface_ammonia_d, 'd_shape')
	GC_surface_ammonia_djf = (GC_surface_ammonia_d+GC_surface_ammonia_jf)/2
	#print (GC_surface_ammonia_djf, 'djf_shape')

	# get GEOS-Chem lon and lat
	gc_lon = GC_surface_ammonia_AM['lon']
	gc_lat = GC_surface_ammonia_AM['lat']

	# get number of sites from size of long and lat:
	nsites=len(sites_lon_c)
	#print (nsites,'nsites')
	# Define GEOS-Chem data obtained at same location as monitoring sites:
	gc_data_ammonia_annual=np.zeros(nsites)
	gc_data_ammonia_msp =np.zeros(nsites)
	gc_data_ammonia_mam=np.zeros(nsites)
	gc_data_ammonia_jja=np.zeros(nsites)
	gc_data_ammonia_son=np.zeros(nsites)
	gc_data_ammonia_djf=np.zeros(nsites)

	#print (len(sites_lat), sites_lat,'(len(sites_lat))')
	#extract GEOS-Chem data using DEFRA sites lat long 
	for w in range(len(sites_lat_c)):
		#print (w, 'w')
		#print ((sites_lat[w]), 'sites_lat[w]')
		# lat and lon indices:
		lon_index = np.argmin(np.abs(np.subtract(sites_lon_c[w],gc_lon)))
		lat_index = np.argmin(np.abs(np.subtract(sites_lat_c[w],gc_lat)))

		#print (lon_index)
		#print (lat_index)
		gc_data_ammonia_annual[w] = GC_surface_ammonia_AM[lon_index, lat_index]
		gc_data_ammonia_msp[w] = GC_surface_ammonia_msp[lon_index, lat_index]
		gc_data_ammonia_mam[w] = GC_surface_ammonia_mam[lon_index, lat_index]
		gc_data_ammonia_jja[w] = GC_surface_ammonia_jja[lon_index, lat_index]
		gc_data_ammonia_son[w] = GC_surface_ammonia_son[lon_index, lat_index]
		gc_data_ammonia_djf[w] = GC_surface_ammonia_djf[lon_index, lat_index]

	# Compare DERFA and GEOS-Chem:
	nmb_Annual=100.*((np.nanmean(gc_data_ammonia_annual))- np.nanmean(sites_ammonia_AM_c))/np.nanmean(sites_ammonia_AM_c)
	nmb_msp=100.*((np.nanmean(gc_data_ammonia_msp))- np.nanmean(sites_ammonia_msp_c))/np.nanmean(sites_ammonia_msp_c)
	nmb_mam=100.*((np.nanmean(gc_data_ammonia_mam))- np.nanmean(sites_ammonia_mam_c))/np.nanmean(sites_ammonia_mam_c)
	nmb_jja=100.*((np.nanmean(gc_data_ammonia_jja))- np.nanmean(sites_ammonia_jja_c))/np.nanmean(sites_ammonia_jja_c)
	nmb_son=100.*((np.nanmean(gc_data_ammonia_son))- np.nanmean(sites_ammonia_son_c))/np.nanmean(sites_ammonia_son_c)
	nmb_djf=100.*((np.nanmean(gc_data_ammonia_djf))- np.nanmean(sites_ammonia_djf_c))/np.nanmean(sites_ammonia_djf_c)
	print(' DEFRA NMB_Annual= ', nmb_Annual)
	print(' DEFRA NMB_msp = ', nmb_msp)
	print(' DEFRA NMB_mam = ', nmb_mam)
	print(' DEFRA NMB_jja = ', nmb_jja)
	print(' DEFRA NMB_son = ', nmb_son)
	print(' DEFRA NMB_djf = ', nmb_djf)


	#correlation
	correlate_Annual=stats.pearsonr(gc_data_ammonia_annual,sites_ammonia_AM_c)

	# dropping nan values and compute correlation
	nas_msp = np.logical_or(np.isnan(gc_data_ammonia_msp), np.isnan(sites_ammonia_msp_c))
	correlate_msp = stats.pearsonr(gc_data_ammonia_msp[~nas_msp],sites_ammonia_msp_c[~nas_msp])


	nas_mam = np.logical_or(np.isnan(gc_data_ammonia_mam), np.isnan(sites_ammonia_mam_c))
	correlate_mam = stats.pearsonr(gc_data_ammonia_mam[~nas_mam],sites_ammonia_mam_c[~nas_mam])

	nas_jja = np.logical_or(np.isnan(gc_data_ammonia_jja), np.isnan(sites_ammonia_jja_c))
	correlate_jja = stats.pearsonr(gc_data_ammonia_jja[~nas_jja],sites_ammonia_jja_c[~nas_jja])

	nas_son = np.logical_or(np.isnan(gc_data_ammonia_son), np.isnan(sites_ammonia_son_c))
	correlate_son = stats.pearsonr(gc_data_ammonia_son[~nas_son],sites_ammonia_son_c[~nas_son])

	nas_djf = np.logical_or(np.isnan(gc_data_ammonia_djf), np.isnan(sites_ammonia_djf_c))
	correlate_djf = stats.pearsonr(gc_data_ammonia_djf[~nas_djf],sites_ammonia_djf_c[~nas_djf])

	#print('Correlation = ',correlate_Annual)

	return GC_surface_ammonia_msp, sites_ammonia_msp

def Ammonium():
	#read UKEAP ammonium datasets here scratch_alok -> /scratch/uptrop/ap744
	path='/scratch/uptrop/ap744/UKEAP_data/UKEAP_NH3_particulate_ammonium/particulate_ammonium/'
	ammonium_files=glob.glob(path + '27-UKA0*-2016_particulate_ammonium_*.csv')
	#print (ammonium_files)

	# read csv file having DEFRA sites details
	sites = pd.read_csv('/scratch/uptrop/ap744/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_AcidGases_Aerosol_sites_details.csv', encoding= 'unicode_escape')
	#print (sites.head(10))
	ID = sites["UK-AIR_ID"]
	#print (ID)

	# site wise annual mean computation  
	x = []
	for f in ammonium_files:
		df = pd.read_csv(f,parse_dates=["Start Date", "End Date"])  
		#print (df.head(5))
		#print (len(ammonium_files))
		sitesA = sites.copy()
		#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

		#Annual Mean calculation
		mean_A= df["Measurement"].mean() # to compute annual mean
		#print (mean_A, f[88:96])
		
		#MAMJJAS mean Calculation
		msp_start = pd.to_datetime("15/02/2016")
		msp_end = pd.to_datetime("15/10/2016")
		msp_subset = df[(df["Start Date"] > msp_start) & (df["End Date"] < msp_end)]
		mean_msp = msp_subset["Measurement"].mean()
		
		#MAM mean Calculation
		mam_start = pd.to_datetime("15/02/2016")
		mam_end = pd.to_datetime("15/06/2016")
		mam_subset = df[(df["Start Date"] > mam_start) & (df["End Date"] < mam_end)]
		mean_mam = mam_subset["Measurement"].mean()
		
		#JJA mean Calculation
		jja_start = pd.to_datetime("15/05/2016")
		jja_end = pd.to_datetime("15/09/2016")
		jja_subset = df[(df["Start Date"] > jja_start) & (df["End Date"] < jja_end)]
		mean_jja = jja_subset["Measurement"].mean()

		#SON mean Calculation
		son_start = pd.to_datetime("15/08/2016")
		son_end = pd.to_datetime("15/11/2016")
		son_subset = df[(df["Start Date"] > son_start) & (df["End Date"] < son_end)]
		mean_son = son_subset["Measurement"].mean()
		
		#DJF mean Calculation
		
		d_start = pd.to_datetime("15/11/2016")
		d_end = pd.to_datetime("31/12/2016")
		d_subset = df[(df["Start Date"] > d_start) & (df["End Date"] < d_end)]
		mean_d = d_subset["Measurement"].mean()
		#print (mean_d, 'mean_d')
		
		
		jf_start = pd.to_datetime("01/01/2016")
		jf_end = pd.to_datetime("15/03/2016")
		jf_subset = df[(df["Start Date"] > jf_start) & (df["End Date"] < jf_end)]
		mean_jf = jf_subset["Measurement"].mean()
		#print (mean_jf, 'mean_jf')
		
		
		mean_djf_a  = np.array([mean_d, mean_jf])
		
		mean_djf = np.nanmean(mean_djf_a, axis=0)
		#print (mean_djf, 'mean_djf')
		
		sitesA["ammonium_annual_mean"] = mean_A
		sitesA["ammonium_msp_mean"] = mean_msp
		sitesA["ammonium_mam_mean"] = mean_mam
		sitesA["ammonium_jja_mean"] = mean_jja
		sitesA["ammonium_son_mean"] = mean_son
		sitesA["ammonium_djf_mean"] = mean_djf
		#print (sitesA.head(10))
		
		x.append(
		{
			'UK-AIR_ID':f[88:96],
			'ammonium_annual_mean':mean_A,
			'ammonium_msp_mean':mean_mam,
			'ammonium_mam_mean':mean_mam,
			'ammonium_jja_mean':mean_jja,
			'ammonium_son_mean':mean_son,
			'ammonium_djf_mean':mean_djf
			}
			)
		#print (x)
		
	id_mean = pd.DataFrame(x)
	#print (id_mean.head(3))

	df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
	#print (df_merge_col.head(32))

	#####export csv file having site wise annual mean information if needed 
	#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/ammonium_annual_mean.csv')

	#drop extra information from pandas dataframe
	df_merge_colA = df_merge_col.drop(['S No','2016_Data'], axis=1)
	df_merge_colA = df_merge_colA.drop(df_merge_colA.index[[2]])
	df_merge_colA.reset_index(drop=True, inplace=True)
	#print (df_merge_colA.head(35))

	df_merge_colB = df_merge_colA.copy()

	###################################################################################
	###########  Delete Data over Scotland           ##################################
	###################################################################################
	df_merge_colB.drop(df_merge_colB[df_merge_colB['Lat'] > 56].index, inplace = True) 
	#print(df_merge_colB.head(11)) 
	df_merge_colB.reset_index(drop=True, inplace=True)
	#print(df_merge_colB.head(11)) 


	# change datatype to float to remove any further problems
	df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
	df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)
	df_merge_colB['Long'] = df_merge_colB['Long'].astype(float)
	df_merge_colB['Lat'] = df_merge_colB['Lat'].astype(float)

	#get sites information
	sites_lon = df_merge_colA['Long']
	sites_lat = df_merge_colA['Lat']


	#get sites information for calculation
	sites_lon_c = df_merge_colB['Long']
	sites_lat_c = df_merge_colB['Lat']

	#getting annual mean data
	sites_ammonium_AM = df_merge_colA['ammonium_annual_mean']
	#seasonal mean data
	sites_ammonium_msp = df_merge_colA['ammonium_msp_mean']
	sites_ammonium_mam = df_merge_colA['ammonium_mam_mean']
	sites_ammonium_jja = df_merge_colA['ammonium_jja_mean']
	sites_ammonium_son = df_merge_colA['ammonium_son_mean']
	sites_ammonium_djf = df_merge_colA['ammonium_djf_mean']
	sites_name = df_merge_colA['Site_Name']
	#print (sites_ammonium_AM, sites_name, sites_lat, sites_lon)


	#seasonal mean data for calculation
	sites_ammonium_AM_c = df_merge_colB['ammonium_annual_mean']
	sites_ammonium_msp_c = df_merge_colB['ammonium_msp_mean']
	sites_ammonium_mam_c = df_merge_colB['ammonium_mam_mean']
	sites_ammonium_jja_c = df_merge_colB['ammonium_jja_mean']
	sites_ammonium_son_c = df_merge_colB['ammonium_son_mean']
	sites_ammonium_djf_c = df_merge_colB['ammonium_djf_mean']
	sites_name_c = df_merge_colB['Site_Name']



	##############  new to read files  #############
	#####Reading GEOS-Chem files ################
	path_AerosolMass_2 = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/AerosolMass/2016/"
	########################### 50% increase in NH3 Emission ##################################
	path_AerosolMass_50increase = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_scale_nh3_emis/AerosolMass/2016/"

	os.chdir(path_AerosolMass_2)
	Aerosols = sorted(glob.glob("GEOSChem.AerosolMass*nc4"))

	Aerosols = Aerosols[:]
	Aerosols = [xr.open_dataset(file) for file in Aerosols]



	GC_surface_ammonium = [data['AerMassNH4'].isel(time=0,lev=0) for data in Aerosols]
	#print (GC_surface_ammonium)

	#Geos-Chem Annual Mean
	GC_surface_ammonium_AM = sum(GC_surface_ammonium)/len(GC_surface_ammonium)
	#print (GC_surface_ammonium_AM,'AnnualMean')
	#print (GC_surface_ammonium_AM.shape,'AnnualMean shape')

	#Geos-Chem seasonal Mean

	GC_surface_ammonium_msp = sum(GC_surface_ammonium[2:9])/len(GC_surface_ammonium[2:9])
	GC_surface_ammonium_mam = sum(GC_surface_ammonium[2:5])/len(GC_surface_ammonium[2:5])
	#print (GC_surface_ammonium_mam.shape, 'MAM-shape')

	GC_surface_ammonium_jja = sum(GC_surface_ammonium[5:8])/len(GC_surface_ammonium[5:8])
	#print (GC_surface_ammonium_jja)

	GC_surface_ammonium_son = sum(GC_surface_ammonium[8:11])/len(GC_surface_ammonium[8:11])
	#print (GC_surface_ammonium_son)

	GC_surface_ammonium_jf = sum(GC_surface_ammonium[0:2])/len(GC_surface_ammonium[0:2])
	#print (GC_surface_ammonium_jf, 'jf_shape')

	GC_surface_ammonium_d = GC_surface_ammonium[11]
	#print (GC_surface_ammonium_d, 'd_shape')

	#mean of JF and Dec using np.array --> creating problem in plotting
	#GC_surface_ammonium_djf_a = np.array([GC_surface_ammonium_jf,GC_surface_ammonium_d])
	#GC_surface_ammonium_djf = np.nanmean(GC_surface_ammonium_djf_a,axis=0)
	#print (GC_surface_ammonium_djf, 'djf_shape')


	GC_surface_ammonium_djf = (GC_surface_ammonium_d+GC_surface_ammonium_jf)/2
	#print (GC_surface_ammonium_djf, 'djf_shape')

	#GEOS-Chem lat long information --Not working properly
	#gc_lon = Aerosols[0]['lon']
	#gc_lat = Aerosols[0]['lat']
	#gc_lon,gc_lat = np.meshgrid(gc_lon,gc_lat)

	# get GEOS-Chem lon and lat
	gc_lon = GC_surface_ammonium_AM['lon']
	gc_lat = GC_surface_ammonium_AM['lat']


	# get number of sites from size of long and lat:
	nsites=len(sites_lon_c)

	# Define GEOS-Chem data obtained at same location as monitoring sites:
	gc_data_ammonium_annual=np.zeros(nsites)
	gc_data_ammonium_msp=np.zeros(nsites)
	gc_data_ammonium_mam=np.zeros(nsites)
	gc_data_ammonium_jja=np.zeros(nsites)
	gc_data_ammonium_son=np.zeros(nsites)
	gc_data_ammonium_djf=np.zeros(nsites)


	#extract GEOS-Chem data using DEFRA sites lat long 
	for w in range(len(sites_lat_c)):
		#print ((sites_lat[w],gc_lat))
		# lat and lon indices:
		lon_index = np.argmin(np.abs(np.subtract(sites_lon_c[w],gc_lon)))
		lat_index = np.argmin(np.abs(np.subtract(sites_lat_c[w],gc_lat)))

		#print (lon_index)
		#print (lat_index)
		gc_data_ammonium_annual[w] = GC_surface_ammonium_AM[lon_index, lat_index]
		gc_data_ammonium_msp[w] = GC_surface_ammonium_msp[lon_index, lat_index]
		gc_data_ammonium_mam[w] = GC_surface_ammonium_mam[lon_index, lat_index]
		gc_data_ammonium_jja[w] = GC_surface_ammonium_jja[lon_index, lat_index]
		gc_data_ammonium_son[w] = GC_surface_ammonium_son[lon_index, lat_index]
		gc_data_ammonium_djf[w] = GC_surface_ammonium_djf[lon_index, lat_index]

	#print (gc_data_ammonium_annual.shape)
	#print (sites_ammonium_AM.shape)

	# quick scatter plot
	#plt.plot(sites_ammonium_AM,gc_data_ammonium_annual,'o')
	#plt.show()

	# Compare DERFA and GEOS-Chem:
	#Normalized mean bias
	nmb_Annual=100.*((np.nanmean(gc_data_ammonium_annual))- np.nanmean(sites_ammonium_AM_c))/np.nanmean(sites_ammonium_AM_c)
	nmb_msp=100.*((np.nanmean(gc_data_ammonium_msp))- np.nanmean(sites_ammonium_msp_c))/np.nanmean(sites_ammonium_msp_c)
	nmb_mam=100.*((np.nanmean(gc_data_ammonium_mam))- np.nanmean(sites_ammonium_mam_c))/np.nanmean(sites_ammonium_mam_c)
	nmb_jja=100.*((np.nanmean(gc_data_ammonium_jja))- np.nanmean(sites_ammonium_jja_c))/np.nanmean(sites_ammonium_jja_c)
	nmb_son=100.*((np.nanmean(gc_data_ammonium_son))- np.nanmean(sites_ammonium_son_c))/np.nanmean(sites_ammonium_son_c)
	nmb_djf=100.*((np.nanmean(gc_data_ammonium_djf))- np.nanmean(sites_ammonium_djf_c))/np.nanmean(sites_ammonium_djf_c)

	print(' DEFRA NMB_Annual= ', nmb_Annual)
	print(' DEFRA NMB_msp = ', nmb_msp)
	print(' DEFRA NMB_mam = ', nmb_mam)
	print(' DEFRA NMB_jja = ', nmb_jja)
	print(' DEFRA NMB_son = ', nmb_son)
	print(' DEFRA NMB_djf = ', nmb_djf)
	#correlation
	correlate_Annual=stats.pearsonr(gc_data_ammonium_annual,sites_ammonium_AM_c)

	# dropping nan values and compute correlation
	nas_msp = np.logical_or(np.isnan(gc_data_ammonium_msp), np.isnan(sites_ammonium_msp_c))
	correlate_msp = stats.pearsonr(gc_data_ammonium_msp[~nas_msp],sites_ammonium_msp_c[~nas_msp])

	nas_mam = np.logical_or(np.isnan(gc_data_ammonium_mam), np.isnan(sites_ammonium_mam_c))
	correlate_mam = stats.pearsonr(gc_data_ammonium_mam[~nas_mam],sites_ammonium_mam_c[~nas_mam])

	nas_jja = np.logical_or(np.isnan(gc_data_ammonium_jja), np.isnan(sites_ammonium_jja_c))
	correlate_jja = stats.pearsonr(gc_data_ammonium_jja[~nas_jja],sites_ammonium_jja_c[~nas_jja])

	nas_son = np.logical_or(np.isnan(gc_data_ammonium_son), np.isnan(sites_ammonium_son_c))
	correlate_son = stats.pearsonr(gc_data_ammonium_son[~nas_son],sites_ammonium_son_c[~nas_son])

	nas_djf = np.logical_or(np.isnan(gc_data_ammonium_djf), np.isnan(sites_ammonium_djf_c))
	correlate_djf = stats.pearsonr(gc_data_ammonium_djf[~nas_djf],sites_ammonium_djf_c[~nas_djf])

	#print('Correlation = ',correlate_Annual)

	return GC_surface_ammonium_msp, sites_ammonium_msp

def HNO3():
	#read UKEAP HNO3 datasets here scratch_alok -> /scratch/uptrop/ap744
	path='/scratch/uptrop/ap744/UKEAP_data/UKEAP_AcidGases_Aerosol/UKEAP_HNO3/'
	HNO3_files=glob.glob(path + '28-UKA0*-2016_*.csv')
	#print (HNO3_files)

	# read csv file having DEFRA sites details
	sites = pd.read_csv('/scratch/uptrop/ap744/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_NH3_sites_details.csv', encoding= 'unicode_escape')
	#print (sites.head(10))
	ID = sites["UK-AIR_ID"]
	#print (ID)

	# site wise annual mean computation  
	x = []
	for f in HNO3_files:
		df = pd.read_csv(f,parse_dates=["Start Date", "End Date"])  
		#print (df.head(5))
		#print (len(HNO3_files))
		sitesA = sites.copy()
		#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

		#Annual Mean calculation
		mean_A= df["Measurement"].mean() # to compute annual mean
		#print (mean_A, f[71:79])
		
		#MAMJJAS mean Calculation
		msp_start = pd.to_datetime("15/02/2016")
		msp_end = pd.to_datetime("15/10/2016")
		msp_subset = df[(df["Start Date"] > msp_start) & (df["End Date"] < msp_end)]
		mean_msp = msp_subset["Measurement"].mean()
		
		#MAM mean Calculation
		mam_start = pd.to_datetime("15/02/2016")
		mam_end = pd.to_datetime("15/06/2016")
		mam_subset = df[(df["Start Date"] > mam_start) & (df["End Date"] < mam_end)]
		mean_mam = mam_subset["Measurement"].mean()
		
		#JJA mean Calculation
		jja_start = pd.to_datetime("15/05/2016")
		jja_end = pd.to_datetime("15/09/2016")
		jja_subset = df[(df["Start Date"] > jja_start) & (df["End Date"] < jja_end)]
		mean_jja = jja_subset["Measurement"].mean()

		#SON mean Calculation
		son_start = pd.to_datetime("15/08/2016")
		son_end = pd.to_datetime("15/11/2016")
		son_subset = df[(df["Start Date"] > son_start) & (df["End Date"] < son_end)]
		mean_son = son_subset["Measurement"].mean()
		
		#DJF mean Calculation
		
		d_start = pd.to_datetime("15/11/2016")
		d_end = pd.to_datetime("31/12/2016")
		d_subset = df[(df["Start Date"] > d_start) & (df["End Date"] < d_end)]
		mean_d = d_subset["Measurement"].mean()
		#print (mean_d, 'mean_d')
		
		
		jf_start = pd.to_datetime("01/01/2016")
		jf_end = pd.to_datetime("15/03/2016")
		jf_subset = df[(df["Start Date"] > jf_start) & (df["End Date"] < jf_end)]
		mean_jf = jf_subset["Measurement"].mean()
		#print (mean_jf, 'mean_jf')
		
		
		mean_djf_a  = np.array([mean_d, mean_jf])
		
		mean_djf = np.nanmean(mean_djf_a, axis=0)
		#print (mean_djf, 'mean_djf')
		
		sitesA["HNO3_annual_mean"] = mean_A
		sitesA["HNO3_msp_mean"] = mean_msp
		sitesA["HNO3_mam_mean"] = mean_mam
		sitesA["HNO3_jja_mean"] = mean_jja
		sitesA["HNO3_son_mean"] = mean_son
		sitesA["HNO3_djf_mean"] = mean_djf
		#print (sitesA.head(50))
		
		x.append(
		{
			'UK-AIR_ID':f[71:79],
			'HNO3_annual_mean':mean_A,
			'HNO3_msp_mean':mean_msp,
			'HNO3_mam_mean':mean_mam,
			'HNO3_jja_mean':mean_jja,
			'HNO3_son_mean':mean_son,
			'HNO3_djf_mean':mean_djf
			}
			)
		#print (x)
		
	id_mean = pd.DataFrame(x)
	#print (id_mean.head(3))

	df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
	#print (df_merge_col.head(50))

	#####export csv file having site wise annual mean information if needed 
	#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/HNO3_annual_mean.csv')

	#drop extra information from pandas dataframe
	df_merge_colA = df_merge_col.drop(['S No'], axis=1)
	df_merge_colA = df_merge_colA.drop(df_merge_colA.index[[2,7,9,24]])
	df_merge_colA.reset_index(drop=True, inplace=True)
	#print (df_merge_colA.head(50))

	df_merge_colB = df_merge_colA.copy()

	###################################################################################
	###########  Delete Data over Scotland           ##################################
	###################################################################################
	df_merge_colB.drop(df_merge_colB[df_merge_colB['Lat'] > 56].index, inplace = True) 
	#print(df_merge_colB.head(11)) 
	df_merge_colB.reset_index(drop=True, inplace=True)
	#print(df_merge_colB.head(11)) 


	# change datatype to float to remove any further problems
	df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
	df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)
	df_merge_colB['Long'] = df_merge_colB['Long'].astype(float)
	df_merge_colB['Lat'] = df_merge_colB['Lat'].astype(float)

	#get sites information
	sites_lon = df_merge_colA['Long']
	sites_lat = df_merge_colA['Lat']


	#get sites information for calculation
	sites_lon_c = df_merge_colB['Long']
	sites_lat_c = df_merge_colB['Lat']

	#getting annual mean data
	sites_HNO3_AM = df_merge_colA['HNO3_annual_mean']
	#seasonal mean data
	sites_HNO3_msp = df_merge_colA['HNO3_msp_mean']
	sites_HNO3_mam = df_merge_colA['HNO3_mam_mean']
	sites_HNO3_jja = df_merge_colA['HNO3_jja_mean']
	sites_HNO3_son = df_merge_colA['HNO3_son_mean']
	sites_HNO3_djf = df_merge_colA['HNO3_djf_mean']
	sites_name = df_merge_colA['Site_Name']
	#print (sites_HNO3_AM, sites_name, sites_lat, sites_lon)


	#seasonal mean data for calculation
	sites_HNO3_AM_c = df_merge_colB['HNO3_annual_mean']
	sites_HNO3_msp_c = df_merge_colB['HNO3_msp_mean']
	sites_HNO3_mam_c = df_merge_colB['HNO3_mam_mean']
	sites_HNO3_jja_c = df_merge_colB['HNO3_jja_mean']
	sites_HNO3_son_c = df_merge_colB['HNO3_son_mean']
	sites_HNO3_djf_c = df_merge_colB['HNO3_djf_mean']
	sites_name_c = df_merge_colB['Site_Name']



	#########################       Reading GEOS-Chem files    ################################
	Species = sorted(glob.glob("/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/SpeciesConc/2016/GEOSChem.SpeciesConc*.nc4"))  # iccw
	#print (Species)
	########################### 50% increase in NH3 Emission ##################################
	#Species = sorted(glob.glob("/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_scale_nh3_emis/SpeciesConc/2016/GEOSChem.SpeciesConc*.nc4"))  #scale Nh3 by 50%

	StateMet = sorted(glob.glob("/scratch/uptrop/ap744/GEOS-Chem_outputs/GEOSChem.StateMet.2016*b.nc4"))


	Species = Species[:] 
	StateMet = StateMet[:]

	#print(Species, StateMet, sep = "\n")

	Species  = [xr.open_dataset(file) for file in Species]
	StateMet = [xr.open_dataset(file) for file in StateMet]



	#ds = xr.open_mfdataset(StateMet)
	#monthly_data = ds.resample(time='m').mean()
	#print(monthly_data)
	#monthly_data_StateMet = StateMet.resample(freq = 'm', dim = 'time', how = 'mean')
	#print(monthly_data_StateMet)

	#HNO3 sufrace layer
	GC_surface_HNO3 = [data['SpeciesConc_HNO3'].isel(time=0,lev=0) for data in Species]
	#print (GC_surface_HNO3)

	#Avogadro's number [mol-1]
	AVOGADRO = 6.022140857e+23

	# Typical molar mass of air [kg mol-1]
	MW_AIR = 28.9644e-3
	# convert unit for HNO3 (dry mol/mol to ug/m3)
	surface_AIRDEN = [data['Met_AIRDEN'].isel(time=0,lev=0) for data in StateMet] #kg/m3
	surface_AIRNUMDEN_a = np.asarray(surface_AIRDEN)/MW_AIR #mol/m3
	surface_AIRNUMDEN_b = surface_AIRNUMDEN_a*AVOGADRO # unit molec air/m3
	surface_AIRNUMDEN = surface_AIRNUMDEN_b/1e6 #unit molec air/cm3

	surface_HNO3_mass  = [x*y*63/(6.022*1e11) for (x,y) in zip(GC_surface_HNO3,surface_AIRNUMDEN)]
	#print (surface_HNO3_mass)

	#Geos-Chem Annual Mean
	GC_surface_HNO3_AM = sum(surface_HNO3_mass)/len(surface_HNO3_mass)
	#print (GC_surface_HNO3_AM,'AnnualMean')
	#print (GC_surface_HNO3_AM.shape,'AnnualMean shape')

	#Geos-Chem seasonal Mean
	GC_surface_HNO3_msp = sum(surface_HNO3_mass[2:9])/len(surface_HNO3_mass[2:9])
	GC_surface_HNO3_mam = sum(surface_HNO3_mass[2:5])/len(surface_HNO3_mass[2:5])
	#print (GC_surface_HNO3_mam.shape, 'MAM-shape')

	GC_surface_HNO3_jja = sum(surface_HNO3_mass[5:8])/len(surface_HNO3_mass[5:8])
	#print (GC_surface_HNO3_jja)

	GC_surface_HNO3_son = sum(surface_HNO3_mass[8:11])/len(surface_HNO3_mass[8:11])
	#print (GC_surface_HNO3_son)

	GC_surface_HNO3_jf = sum(surface_HNO3_mass[0:2])/len(surface_HNO3_mass[0:2])
	#print (GC_surface_HNO3_jf, 'jf_shape')

	GC_surface_HNO3_d = surface_HNO3_mass[11]
	#print (GC_surface_HNO3_d, 'd_shape')

	#mean of JF and Dec using np.array --> creating problem in plotting
	#GC_surface_HNO3_djf_a = np.array([GC_surface_HNO3_jf,GC_surface_HNO3_d])
	#GC_surface_HNO3_djf = np.nanmean(GC_surface_HNO3_djf_a,axis=0)
	#print (GC_surface_HNO3_djf, 'djf_shape')


	GC_surface_HNO3_djf = (GC_surface_HNO3_d+GC_surface_HNO3_jf)/2
	#print (GC_surface_HNO3_djf, 'djf_shape')

	#GEOS-Chem lat long information --Not working properly
	#gc_lon = Aerosols[0]['lon']
	#gc_lat = Aerosols[0]['lat']
	#gc_lon,gc_lat = np.meshgrid(gc_lon,gc_lat)

	# get GEOS-Chem lon and lat
	gc_lon = GC_surface_HNO3_AM['lon']
	gc_lat = GC_surface_HNO3_AM['lat']
	#print (len(gc_lon))
	#print (len(gc_lat))
	#print ((gc_lon))
	#print ((gc_lat))

	# get number of sites from size of long and lat:
	nsites=len(sites_lon_c)

	# Define GEOS-Chem data obtained at same location as monitoring sites:
	gc_data_HNO3_annual=np.zeros(nsites)
	gc_data_HNO3_msp=np.zeros(nsites)
	gc_data_HNO3_mam=np.zeros(nsites)
	gc_data_HNO3_jja=np.zeros(nsites)
	gc_data_HNO3_son=np.zeros(nsites)
	gc_data_HNO3_djf=np.zeros(nsites)


	#extract GEOS-Chem data using DEFRA sites lat long 
	for w in range(len(sites_lat_c)):
		#print ((sites_lat[w],gc_lat))
		# lat and lon indices:
		lon_index = np.argmin(np.abs(np.subtract(sites_lon_c[w],gc_lon)))
		lat_index = np.argmin(np.abs(np.subtract(sites_lat_c[w],gc_lat)))

		#print (lon_index)
		#print (lat_index)
		gc_data_HNO3_annual[w] = GC_surface_HNO3_AM[lon_index, lat_index]
		gc_data_HNO3_msp[w] = GC_surface_HNO3_msp[lon_index, lat_index]
		gc_data_HNO3_mam[w] = GC_surface_HNO3_mam[lon_index, lat_index]
		gc_data_HNO3_jja[w] = GC_surface_HNO3_jja[lon_index, lat_index]
		gc_data_HNO3_son[w] = GC_surface_HNO3_son[lon_index, lat_index]
		gc_data_HNO3_djf[w] = GC_surface_HNO3_djf[lon_index, lat_index]

	#print (gc_data_HNO3_annual.shape)
	#print (sites_HNO3_AM.shape)

	# quick scatter plot
	#plt.plot(sites_HNO3_AM,gc_data_HNO3_annual,'o')
	#plt.show()

	# Compare DERFA and GEOS-Chem:
	#Normalized mean bias

	nmb_Annual=100.*((np.nanmean(gc_data_HNO3_annual))- np.nanmean(sites_HNO3_AM_c))/np.nanmean(sites_HNO3_AM_c)
	nmb_msp=100.*((np.nanmean(gc_data_HNO3_msp))- np.nanmean(sites_HNO3_msp_c))/np.nanmean(sites_HNO3_msp_c)
	nmb_mam=100.*((np.nanmean(gc_data_HNO3_mam))- np.nanmean(sites_HNO3_mam_c))/np.nanmean(sites_HNO3_mam_c)
	nmb_jja=100.*((np.nanmean(gc_data_HNO3_jja))- np.nanmean(sites_HNO3_jja_c))/np.nanmean(sites_HNO3_jja_c)
	nmb_son=100.*((np.nanmean(gc_data_HNO3_son))- np.nanmean(sites_HNO3_son_c))/np.nanmean(sites_HNO3_son_c)
	nmb_djf=100.*((np.nanmean(gc_data_HNO3_djf))- np.nanmean(sites_HNO3_djf_c))/np.nanmean(sites_HNO3_djf_c)


	print(' DEFRA NMB_Annual= ', nmb_Annual)
	print(' DEFRA NMB_msp = ', nmb_msp)
	print(' DEFRA NMB_mam = ', nmb_mam)
	print(' DEFRA NMB_jja = ', nmb_jja)
	print(' DEFRA NMB_son = ', nmb_son)
	print(' DEFRA NMB_djf = ', nmb_djf)


	#correlation
	correlate_Annual=stats.pearsonr(gc_data_HNO3_annual,sites_HNO3_AM_c)

	# dropping nan values and compute correlation
	nas_msp = np.logical_or(np.isnan(gc_data_HNO3_msp), np.isnan(sites_HNO3_msp_c))
	correlate_msp = stats.pearsonr(gc_data_HNO3_msp[~nas_msp],sites_HNO3_msp_c[~nas_msp])

	nas_mam = np.logical_or(np.isnan(gc_data_HNO3_mam), np.isnan(sites_HNO3_mam_c))
	correlate_mam = stats.pearsonr(gc_data_HNO3_mam[~nas_mam],sites_HNO3_mam_c[~nas_mam])

	nas_jja = np.logical_or(np.isnan(gc_data_HNO3_jja), np.isnan(sites_HNO3_jja_c))
	correlate_jja = stats.pearsonr(gc_data_HNO3_jja[~nas_jja],sites_HNO3_jja_c[~nas_jja])

	nas_son = np.logical_or(np.isnan(gc_data_HNO3_son), np.isnan(sites_HNO3_son_c))
	correlate_son = stats.pearsonr(gc_data_HNO3_son[~nas_son],sites_HNO3_son_c[~nas_son])

	nas_djf = np.logical_or(np.isnan(gc_data_HNO3_djf), np.isnan(sites_HNO3_djf_c))
	correlate_djf = stats.pearsonr(gc_data_HNO3_djf[~nas_djf],sites_HNO3_djf_c[~nas_djf])

	#print('Correlation = ',correlate_Annual)

	# plotting spatial map model and DEFRA network 
	os.chdir('/home/a/ap744/scratch_alok/shapefiles/GBP_shapefile')
	Europe_shape = r'GBR_adm1.shp'
	Europe_map = ShapelyFeature(Reader(Europe_shape).geometries(),
								   ccrs.PlateCarree(), edgecolor='black',facecolor='none')
	#print ('Shapefile_read')

	return GC_surface_HNO3_msp, sites_HNO3_msp
	
def nitrate():
	#read UKEAP nitrate datasets here scratch_alok -> /scratch/uptrop/ap744
	path='/scratch/uptrop/ap744/UKEAP_data/UKEAP_AcidGases_Aerosol/UKEAP_particulate_nitrate/'
	nitrate_files=glob.glob(path + '28-UKA0*-2016_particulate_nitrate_*.csv')
	#print (nitrate_files)

	# read csv file having DEFRA sites details
	sites = pd.read_csv('/scratch/uptrop/ap744/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_AcidGases_Aerosol_sites_details.csv', encoding= 'unicode_escape')
	#print (sites.head(10))
	ID = sites["UK-AIR_ID"]
	#print (ID)

	# site wise annual mean computation  
	x = []
	for f in nitrate_files:
		df = pd.read_csv(f,parse_dates=["Start Date", "End Date"])  
		#print (df.head(5))
		#print (len(nitrate_files))
		sitesA = sites.copy()
		#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

		#Annual Mean calculation
		mean_A= df["Measurement"].mean() # to compute annual mean
		#print (mean_A, f[86:94])
		
		#MAMJJAS mean Calculation
		msp_start = pd.to_datetime("15/02/2016")
		msp_end = pd.to_datetime("15/10/2016")
		msp_subset = df[(df["Start Date"] > msp_start) & (df["End Date"] < msp_end)]
		mean_msp = msp_subset["Measurement"].mean()
			
		#MAM mean Calculation
		mam_start = pd.to_datetime("15/02/2016")
		mam_end = pd.to_datetime("15/06/2016")
		mam_subset = df[(df["Start Date"] > mam_start) & (df["End Date"] < mam_end)]
		mean_mam = mam_subset["Measurement"].mean()
		
		#JJA mean Calculation
		jja_start = pd.to_datetime("15/05/2016")
		jja_end = pd.to_datetime("15/09/2016")
		jja_subset = df[(df["Start Date"] > jja_start) & (df["End Date"] < jja_end)]
		mean_jja = jja_subset["Measurement"].mean()

		#SON mean Calculation
		son_start = pd.to_datetime("15/08/2016")
		son_end = pd.to_datetime("15/11/2016")
		son_subset = df[(df["Start Date"] > son_start) & (df["End Date"] < son_end)]
		mean_son = son_subset["Measurement"].mean()
		
		#DJF mean Calculation
		
		d_start = pd.to_datetime("15/11/2016")
		d_end = pd.to_datetime("31/12/2016")
		d_subset = df[(df["Start Date"] > d_start) & (df["End Date"] < d_end)]
		mean_d = d_subset["Measurement"].mean()
		#print (mean_d, 'mean_d')
		
		
		jf_start = pd.to_datetime("01/01/2016")
		jf_end = pd.to_datetime("15/03/2016")
		jf_subset = df[(df["Start Date"] > jf_start) & (df["End Date"] < jf_end)]
		mean_jf = jf_subset["Measurement"].mean()
		#print (mean_jf, 'mean_jf')
		
		
		mean_djf_a  = np.array([mean_d, mean_jf])
		
		mean_djf = np.nanmean(mean_djf_a, axis=0)
		#print (mean_djf, 'mean_djf')
		
		sitesA["nitrate_annual_mean"] = mean_A
		sitesA["nitrate_msp_mean"] = mean_msp
		sitesA["nitrate_mam_mean"] = mean_mam
		sitesA["nitrate_jja_mean"] = mean_jja
		sitesA["nitrate_son_mean"] = mean_son
		sitesA["nitrate_djf_mean"] = mean_djf
		#print (sitesA.head(10))
		
		x.append(
		{
			'UK-AIR_ID':f[86:94],
			'nitrate_annual_mean':mean_A,
			'nitrate_msp_mean':mean_msp,
			'nitrate_mam_mean':mean_mam,
			'nitrate_jja_mean':mean_jja,
			'nitrate_son_mean':mean_son,
			'nitrate_djf_mean':mean_djf
			}
			)
		#print (x)
		
	id_mean = pd.DataFrame(x)
	#print (id_mean.head(3))

	df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
	#print (df_merge_col.head(25))

	#####export csv file having site wise annual mean information if needed 
	#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/nitrate_annual_mean.csv')

	#drop extra information from pandas dataframe
	df_merge_colA = df_merge_col.drop(['S No','2016_Data'], axis=1)
	df_merge_colA = df_merge_colA.drop(df_merge_colA.index[[2,7,9,24]])
	df_merge_colA.reset_index(drop=True, inplace=True)
	#print (df_merge_colA.head(50))
	df_merge_colB = df_merge_colA.copy()

	###################################################################################
	###########  Delete Data over Scotland           ##################################
	###################################################################################
	df_merge_colB.drop(df_merge_colB[df_merge_colB['Lat'] > 56].index, inplace = True) 
	#print(df_merge_colB.head(11)) 
	df_merge_colB.reset_index(drop=True, inplace=True)
	#print(df_merge_colB.head(11)) 


	# change datatype to float to remove any further problems
	df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
	df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)
	df_merge_colB['Long'] = df_merge_colB['Long'].astype(float)
	df_merge_colB['Lat'] = df_merge_colB['Lat'].astype(float)

	#get sites information
	sites_lon = df_merge_colA['Long']
	sites_lat = df_merge_colA['Lat']


	#get sites information for calculation
	sites_lon_c = df_merge_colB['Long']
	sites_lat_c = df_merge_colB['Lat']

	#getting annual mean data
	sites_nitrate_AM = df_merge_colA['nitrate_annual_mean']
	#seasonal mean data
	sites_nitrate_msp = df_merge_colA['nitrate_msp_mean']
	sites_nitrate_mam = df_merge_colA['nitrate_mam_mean']
	sites_nitrate_jja = df_merge_colA['nitrate_jja_mean']
	sites_nitrate_son = df_merge_colA['nitrate_son_mean']
	sites_nitrate_djf = df_merge_colA['nitrate_djf_mean']
	sites_name = df_merge_colA['Site_Name']
	#print (sites_nitrate_AM, sites_name, sites_lat, sites_lon)

	#seasonal mean data for calculation
	sites_nitrate_AM_c = df_merge_colB['nitrate_annual_mean']
	sites_nitrate_msp_c = df_merge_colB['nitrate_msp_mean']
	sites_nitrate_mam_c = df_merge_colB['nitrate_mam_mean']
	sites_nitrate_jja_c = df_merge_colB['nitrate_jja_mean']
	sites_nitrate_son_c = df_merge_colB['nitrate_son_mean']
	sites_nitrate_djf_c = df_merge_colB['nitrate_djf_mean']
	sites_name_c = df_merge_colB['Site_Name']

	##############  new to read files  #############
	#####Reading GEOS-Chem files ################
	path_AerosolMass_2 = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/AerosolMass/2016/"

	########################### 50% increase in NH3 Emission ##################################
	path_AerosolMass_50increase = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_scale_nh3_emis/AerosolMass/2016/"

	os.chdir(path_AerosolMass_2)
	Aerosols = sorted(glob.glob("GEOSChem.AerosolMass*nc4"))

	Aerosols = Aerosols[:]
	Aerosols = [xr.open_dataset(file) for file in Aerosols]

	GC_surface_nitrate = [data['AerMassNIT'].isel(time=0,lev=0) for data in Aerosols]
	#print (GC_surface_nitrate)

	#Geos-Chem Annual Mean
	GC_surface_nitrate_AM = sum(GC_surface_nitrate)/len(GC_surface_nitrate)
	#print (GC_surface_nitrate_AM,'AnnualMean')
	#print (GC_surface_nitrate_AM.shape,'AnnualMean shape')

	#Geos-Chem seasonal Mean
	GC_surface_nitrate_msp = sum(GC_surface_nitrate[2:9])/len(GC_surface_nitrate[2:9])
	GC_surface_nitrate_mam = sum(GC_surface_nitrate[2:5])/len(GC_surface_nitrate[2:5])
	#print (GC_surface_nitrate_mam.shape, 'MAM-shape')

	GC_surface_nitrate_jja = sum(GC_surface_nitrate[5:8])/len(GC_surface_nitrate[5:8])
	#print (GC_surface_nitrate_jja)

	GC_surface_nitrate_son = sum(GC_surface_nitrate[8:11])/len(GC_surface_nitrate[8:11])
	#print (GC_surface_nitrate_son)

	GC_surface_nitrate_jf = sum(GC_surface_nitrate[0:2])/len(GC_surface_nitrate[0:2])
	#print (GC_surface_nitrate_jf, 'jf_shape')

	GC_surface_nitrate_d = GC_surface_nitrate[11]
	#print (GC_surface_nitrate_d, 'd_shape')

	#mean of JF and Dec using np.array --> creating problem in plotting
	#GC_surface_nitrate_djf_a = np.array([GC_surface_nitrate_jf,GC_surface_nitrate_d])
	#GC_surface_nitrate_djf = np.nanmean(GC_surface_nitrate_djf_a,axis=0)
	#print (GC_surface_nitrate_djf, 'djf_shape')


	GC_surface_nitrate_djf = (GC_surface_nitrate_d+GC_surface_nitrate_jf)/2
	#print (GC_surface_nitrate_djf, 'djf_shape')

	#GEOS-Chem lat long information --Not working properly
	#gc_lon = Aerosols[0]['lon']
	#gc_lat = Aerosols[0]['lat']
	#gc_lon,gc_lat = np.meshgrid(gc_lon,gc_lat)

	# get GEOS-Chem lon and lat
	gc_lon = GC_surface_nitrate_AM['lon']
	gc_lat = GC_surface_nitrate_AM['lat']
	#print (len(gc_lon))
	#print (len(gc_lat))
	#print ((gc_lon))
	#print ((gc_lat))

	# get number of sites from size of long and lat:
	nsites=len(sites_lon_c)

	# Define GEOS-Chem data obtained at same location as monitoring sites:
	gc_data_nitrate_annual=np.zeros(nsites)
	gc_data_nitrate_msp=np.zeros(nsites)
	gc_data_nitrate_mam=np.zeros(nsites)
	gc_data_nitrate_jja=np.zeros(nsites)
	gc_data_nitrate_son=np.zeros(nsites)
	gc_data_nitrate_djf=np.zeros(nsites)


	#extract GEOS-Chem data using DEFRA sites lat long 
	for w in range(len(sites_lat_c)):
		#print ((sites_lat[w],gc_lat))
		# lat and lon indices:
		lon_index = np.argmin(np.abs(np.subtract(sites_lon_c[w],gc_lon)))
		lat_index = np.argmin(np.abs(np.subtract(sites_lat_c[w],gc_lat)))

		#print (lon_index)
		#print (lat_index)
		gc_data_nitrate_annual[w] = GC_surface_nitrate_AM[lon_index, lat_index]
		gc_data_nitrate_msp[w] = GC_surface_nitrate_msp[lon_index, lat_index]
		gc_data_nitrate_mam[w] = GC_surface_nitrate_mam[lon_index, lat_index]
		gc_data_nitrate_jja[w] = GC_surface_nitrate_jja[lon_index, lat_index]
		gc_data_nitrate_son[w] = GC_surface_nitrate_son[lon_index, lat_index]
		gc_data_nitrate_djf[w] = GC_surface_nitrate_djf[lon_index, lat_index]

	#print (gc_data_nitrate_annual.shape)
	#print (sites_nitrate_AM.shape)

	# quick scatter plot
	#plt.plot(sites_nitrate_AM,gc_data_nitrate_annual,'o')
	#plt.show()

	# Compare DERFA and GEOS-Chem:
	#Normalized mean bias

	nmb_Annual=100.*((np.nanmean(gc_data_nitrate_annual))- np.nanmean(sites_nitrate_AM_c))/np.nanmean(sites_nitrate_AM_c)
	nmb_msp=100.*((np.nanmean(gc_data_nitrate_msp))- np.nanmean(sites_nitrate_msp_c))/np.nanmean(sites_nitrate_msp_c)
	nmb_mam=100.*((np.nanmean(gc_data_nitrate_mam))- np.nanmean(sites_nitrate_mam_c))/np.nanmean(sites_nitrate_mam_c)
	nmb_jja=100.*((np.nanmean(gc_data_nitrate_jja))- np.nanmean(sites_nitrate_jja_c))/np.nanmean(sites_nitrate_jja_c)
	nmb_son=100.*((np.nanmean(gc_data_nitrate_son))- np.nanmean(sites_nitrate_son_c))/np.nanmean(sites_nitrate_son_c)
	nmb_djf=100.*((np.nanmean(gc_data_nitrate_djf))- np.nanmean(sites_nitrate_djf_c))/np.nanmean(sites_nitrate_djf_c)

	print(' DEFRA NMB_Annual= ', nmb_Annual)
	print(' DEFRA NMB_msp = ', nmb_msp)
	print(' DEFRA NMB_mam = ', nmb_mam)
	print(' DEFRA NMB_jja = ', nmb_jja)
	print(' DEFRA NMB_son = ', nmb_son)
	print(' DEFRA NMB_djf = ', nmb_djf)



	#correlation
	correlate_Annual=stats.pearsonr(gc_data_nitrate_annual,sites_nitrate_AM_c)

	# dropping nan values and compute correlation
	nas_msp = np.logical_or(np.isnan(gc_data_nitrate_msp), np.isnan(sites_nitrate_msp_c))
	correlate_msp = stats.pearsonr(gc_data_nitrate_msp[~nas_msp],sites_nitrate_msp_c[~nas_msp])

	nas_mam = np.logical_or(np.isnan(gc_data_nitrate_mam), np.isnan(sites_nitrate_mam_c))
	correlate_mam = stats.pearsonr(gc_data_nitrate_mam[~nas_mam],sites_nitrate_mam_c[~nas_mam])

	nas_jja = np.logical_or(np.isnan(gc_data_nitrate_jja), np.isnan(sites_nitrate_jja_c))
	correlate_jja = stats.pearsonr(gc_data_nitrate_jja[~nas_jja],sites_nitrate_jja_c[~nas_jja])

	nas_son = np.logical_or(np.isnan(gc_data_nitrate_son), np.isnan(sites_nitrate_son_c))
	correlate_son = stats.pearsonr(gc_data_nitrate_son[~nas_son],sites_nitrate_son_c[~nas_son])

	nas_djf = np.logical_or(np.isnan(gc_data_nitrate_djf), np.isnan(sites_nitrate_djf_c))
	correlate_djf = stats.pearsonr(gc_data_nitrate_djf[~nas_djf],sites_nitrate_djf_c[~nas_djf])

	#print('Correlation = ',correlate_Annual)

	return GC_surface_nitrate_msp, sites_nitrate_msp
	
def sulphate():
	#read UKEAP sulphate datasets here scratch_alok -> /scratch/uptrop/ap744
	path='/scratch/uptrop/ap744/UKEAP_data/UKEAP_AcidGases_Aerosol/UKEAP_Particulate_Sulphate/'
	sulphate_files=glob.glob(path + '28-UKA0*-2016_particulate_sulphate_*.csv')
	#print (sulphate_files)

	# read csv file having DEFRA sites details
	sites = pd.read_csv('/scratch/uptrop/ap744/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_AcidGases_Aerosol_sites_details.csv', encoding= 'unicode_escape')
	#print (sites.head(10))
	ID = sites["UK-AIR_ID"]
	#print (ID)

	# site wise annual mean computation  
	x = []
	for f in sulphate_files:
		df = pd.read_csv(f,parse_dates=["Start Date", "End Date"])  
		#print (df.head(5))
		#print (len(sulphate_files))
		sitesA = sites.copy()
		#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

		#Annual Mean calculation
		mean_A= df["Measurement"].mean() # to compute annual mean
		#print (mean_A, f[87:95])
		
		#MAMJJAS mean Calculation
		msp_start = pd.to_datetime("15/02/2016")
		msp_end = pd.to_datetime("15/10/2016")
		msp_subset = df[(df["Start Date"] > msp_start) & (df["End Date"] < msp_end)]
		mean_msp = msp_subset["Measurement"].mean()
		
		#MAM mean Calculation
		mam_start = pd.to_datetime("15/02/2016")
		mam_end = pd.to_datetime("15/06/2016")
		mam_subset = df[(df["Start Date"] > mam_start) & (df["End Date"] < mam_end)]
		mean_mam = mam_subset["Measurement"].mean()
		
		#JJA mean Calculation
		jja_start = pd.to_datetime("15/05/2016")
		jja_end = pd.to_datetime("15/09/2016")
		jja_subset = df[(df["Start Date"] > jja_start) & (df["End Date"] < jja_end)]
		mean_jja = jja_subset["Measurement"].mean()

		#SON mean Calculation
		son_start = pd.to_datetime("15/08/2016")
		son_end = pd.to_datetime("15/11/2016")
		son_subset = df[(df["Start Date"] > son_start) & (df["End Date"] < son_end)]
		mean_son = son_subset["Measurement"].mean()
		
		#DJF mean Calculation
		
		d_start = pd.to_datetime("15/11/2016")
		d_end = pd.to_datetime("31/12/2016")
		d_subset = df[(df["Start Date"] > d_start) & (df["End Date"] < d_end)]
		mean_d = d_subset["Measurement"].mean()
		#print (mean_d, 'mean_d')
		
		
		jf_start = pd.to_datetime("01/01/2016")
		jf_end = pd.to_datetime("15/03/2016")
		jf_subset = df[(df["Start Date"] > jf_start) & (df["End Date"] < jf_end)]
		mean_jf = jf_subset["Measurement"].mean()
		#print (mean_jf, 'mean_jf')
		
		
		mean_djf_a  = np.array([mean_d, mean_jf])
		
		mean_djf = np.nanmean(mean_djf_a, axis=0)
		#print (mean_djf, 'mean_djf')
		
		sitesA["sulphate_annual_mean"] = mean_A
		sitesA["sulphate_msp_mean"] = mean_msp
		sitesA["sulphate_mam_mean"] = mean_mam
		sitesA["sulphate_jja_mean"] = mean_jja
		sitesA["sulphate_son_mean"] = mean_son
		sitesA["sulphate_djf_mean"] = mean_djf
		#print (sitesA.head(10))
		
		x.append(
		{
			'UK-AIR_ID':f[87:95],
			'sulphate_annual_mean':mean_A,
			'sulphate_msp_mean':mean_msp,
			'sulphate_mam_mean':mean_mam,
			'sulphate_jja_mean':mean_jja,
			'sulphate_son_mean':mean_son,
			'sulphate_djf_mean':mean_djf
			}
			)
		#print (x)
		
	id_mean = pd.DataFrame(x)
	#print (id_mean.head(3))

	df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
	#print (df_merge_col.head(25))

	#####export csv file having site wise annual mean information if needed 
	#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/sulphate_annual_mean.csv')

	#drop extra information from pandas dataframe
	df_merge_colA = df_merge_col.drop(['S No','2016_Data'], axis=1)
	df_merge_colA = df_merge_colA.drop(df_merge_colA.index[[2,7,9,24]])
	df_merge_colA.reset_index(drop=True, inplace=True)
	#print (df_merge_colA.head(35))
	df_merge_colB = df_merge_colA.copy()

	###################################################################################
	###########  Delete Data over Scotland           ##################################
	###################################################################################
	df_merge_colB.drop(df_merge_colB[df_merge_colB['Lat'] > 56].index, inplace = True) 
	#print(df_merge_colB.head(11)) 
	df_merge_colB.reset_index(drop=True, inplace=True)
	#print(df_merge_colB.head(11)) 


	# change datatype to float to remove any further problems
	df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
	df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)
	df_merge_colB['Long'] = df_merge_colB['Long'].astype(float)
	df_merge_colB['Lat'] = df_merge_colB['Lat'].astype(float)

	#get sites information
	sites_lon = df_merge_colA['Long']
	sites_lat = df_merge_colA['Lat']


	#get sites information for calculation
	sites_lon_c = df_merge_colB['Long']
	sites_lat_c = df_merge_colB['Lat']

	#getting annual mean data
	sites_sulphate_AM = df_merge_colA['sulphate_annual_mean']
	#seasonal mean data
	sites_sulphate_msp = df_merge_colA['sulphate_msp_mean']
	sites_sulphate_mam = df_merge_colA['sulphate_mam_mean']
	sites_sulphate_jja = df_merge_colA['sulphate_jja_mean']
	sites_sulphate_son = df_merge_colA['sulphate_son_mean']
	sites_sulphate_djf = df_merge_colA['sulphate_djf_mean']
	sites_name = df_merge_colA['Site_Name']
	#print (sites_sulphate_AM, sites_name, sites_lat, sites_lon)


	#seasonal mean data for calculation
	sites_sulphate_AM_c = df_merge_colB['sulphate_annual_mean']
	sites_sulphate_msp_c = df_merge_colB['sulphate_msp_mean']
	sites_sulphate_mam_c = df_merge_colB['sulphate_mam_mean']
	sites_sulphate_jja_c = df_merge_colB['sulphate_jja_mean']
	sites_sulphate_son_c = df_merge_colB['sulphate_son_mean']
	sites_sulphate_djf_c = df_merge_colB['sulphate_djf_mean']
	sites_name_c = df_merge_colB['Site_Name']



	##############  new to read files  #############
	#####Reading GEOS-Chem files ################
	path_AerosolMass_2 = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/AerosolMass/2016/"

	########################### 50% increase in NH3 Emission ##################################
	path_AerosolMass_50increase = "/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_scale_nh3_emis/AerosolMass/2016/"

	os.chdir(path_AerosolMass_2)
	Aerosols = sorted(glob.glob("GEOSChem.AerosolMass*nc4"))

	Aerosols = Aerosols[:]
	Aerosols = [xr.open_dataset(file) for file in Aerosols]


	GC_surface_sulfate = [data['AerMassSO4'].isel(time=0,lev=0) for data in Aerosols]
	#print (GC_surface_sulfate)

	#Geos-Chem Annual Mean
	GC_surface_sulfate_AM = sum(GC_surface_sulfate)/len(GC_surface_sulfate)
	#print (GC_surface_sulfate_AM,'AnnualMean')
	#print (GC_surface_sulfate_AM.shape,'AnnualMean shape')

	#Geos-Chem seasonal Mean
	GC_surface_sulfate_msp = sum(GC_surface_sulfate[2:9])/len(GC_surface_sulfate[2:9])
	GC_surface_sulfate_mam = sum(GC_surface_sulfate[2:5])/len(GC_surface_sulfate[2:5])
	#print (GC_surface_sulfate_mam.shape, 'MAM-shape')

	GC_surface_sulfate_jja = sum(GC_surface_sulfate[5:8])/len(GC_surface_sulfate[5:8])
	#print (GC_surface_sulfate_jja)

	GC_surface_sulfate_son = sum(GC_surface_sulfate[8:11])/len(GC_surface_sulfate[8:11])
	#print (GC_surface_sulfate_son)

	GC_surface_sulfate_jf = sum(GC_surface_sulfate[0:2])/len(GC_surface_sulfate[0:2])
	#print (GC_surface_sulfate_jf, 'jf_shape')

	GC_surface_sulfate_d = GC_surface_sulfate[11]
	#print (GC_surface_sulfate_d, 'd_shape')

	#mean of JF and Dec using np.array --> creating problem in plotting
	#GC_surface_sulfate_djf_a = np.array([GC_surface_sulfate_jf,GC_surface_sulfate_d])
	#GC_surface_sulfate_djf = np.nanmean(GC_surface_sulfate_djf_a,axis=0)
	#print (GC_surface_sulfate_djf, 'djf_shape')


	GC_surface_sulfate_djf = (GC_surface_sulfate_d+GC_surface_sulfate_jf)/2
	#print (GC_surface_sulfate_djf, 'djf_shape')

	# get GEOS-Chem lon and lat
	gc_lon = GC_surface_sulfate_AM['lon']
	gc_lat = GC_surface_sulfate_AM['lat']
	#print (len(gc_lon))
	#print (len(gc_lat))
	#print ((gc_lon))
	#print ((gc_lat))

	# get number of sites from size of long and lat:
	nsites=len(sites_lon_c)

	# Define GEOS-Chem data obtained at same location as monitoring sites:
	gc_data_sulphate_annual=np.zeros(nsites)
	gc_data_sulphate_msp=np.zeros(nsites)
	gc_data_sulphate_mam=np.zeros(nsites)
	gc_data_sulphate_jja=np.zeros(nsites)
	gc_data_sulphate_son=np.zeros(nsites)
	gc_data_sulphate_djf=np.zeros(nsites)


	#extract GEOS-Chem data using DEFRA sites lat long 
	for w in range(len(sites_lat_c)):
		#print ((sites_lat[w],gc_lat))
		# lat and lon indices:
		lon_index = np.argmin(np.abs(np.subtract(sites_lon_c[w],gc_lon)))
		lat_index = np.argmin(np.abs(np.subtract(sites_lat_c[w],gc_lat)))

		#print (lon_index)
		#print (lat_index)
		gc_data_sulphate_annual[w] = GC_surface_sulfate_AM[lon_index, lat_index]
		gc_data_sulphate_msp[w] = GC_surface_sulfate_mam[lon_index, lat_index]
		gc_data_sulphate_mam[w] = GC_surface_sulfate_mam[lon_index, lat_index]
		gc_data_sulphate_jja[w] = GC_surface_sulfate_jja[lon_index, lat_index]
		gc_data_sulphate_son[w] = GC_surface_sulfate_son[lon_index, lat_index]
		gc_data_sulphate_djf[w] = GC_surface_sulfate_djf[lon_index, lat_index]

	#print (gc_data_sulphate_annual.shape)
	#print (sites_sulphate_AM.shape)

	# quick scatter plot
	#plt.plot(sites_sulphate_AM,gc_data_sulphate_annual,'o')
	#plt.show()

	# Compare DEFRA and GEOS-Chem:
	#Normalized mean bias
	nmb_Annual=100.*((np.nanmean(gc_data_sulphate_annual))- np.nanmean(sites_sulphate_AM_c))/np.nanmean(sites_sulphate_AM_c)
	nmb_msp=100.*((np.nanmean(gc_data_sulphate_msp))- np.nanmean(sites_sulphate_msp_c))/np.nanmean(sites_sulphate_msp_c)
	nmb_mam=100.*((np.nanmean(gc_data_sulphate_mam))- np.nanmean(sites_sulphate_mam_c))/np.nanmean(sites_sulphate_mam_c)
	nmb_jja=100.*((np.nanmean(gc_data_sulphate_jja))- np.nanmean(sites_sulphate_jja_c))/np.nanmean(sites_sulphate_jja_c)
	nmb_son=100.*((np.nanmean(gc_data_sulphate_son))- np.nanmean(sites_sulphate_son_c))/np.nanmean(sites_sulphate_son_c)
	nmb_djf=100.*((np.nanmean(gc_data_sulphate_djf))- np.nanmean(sites_sulphate_djf_c))/np.nanmean(sites_sulphate_djf_c)
	print(' DEFRA NMB_Annual= ', nmb_Annual)
	print(' DEFRA NMB_mam = ', nmb_mam)
	print(' DEFRA NMB_jja = ', nmb_jja)
	print(' DEFRA NMB_son = ', nmb_son)
	print(' DEFRA NMB_djf = ', nmb_djf)

	#correlation
	correlate_Annual=stats.pearsonr(gc_data_sulphate_annual,sites_sulphate_AM_c)

	# dropping nan values and compute correlation
	nas_msp = np.logical_or(np.isnan(gc_data_sulphate_msp), np.isnan(sites_sulphate_msp_c))
	correlate_msp = stats.pearsonr(gc_data_sulphate_msp[~nas_msp],sites_sulphate_msp_c[~nas_msp])

	nas_mam = np.logical_or(np.isnan(gc_data_sulphate_mam), np.isnan(sites_sulphate_mam_c))
	correlate_mam = stats.pearsonr(gc_data_sulphate_mam[~nas_mam],sites_sulphate_mam_c[~nas_mam])

	nas_jja = np.logical_or(np.isnan(gc_data_sulphate_jja), np.isnan(sites_sulphate_jja_c))
	correlate_jja = stats.pearsonr(gc_data_sulphate_jja[~nas_jja],sites_sulphate_jja_c[~nas_jja])

	nas_son = np.logical_or(np.isnan(gc_data_sulphate_son), np.isnan(sites_sulphate_son_c))
	correlate_son = stats.pearsonr(gc_data_sulphate_son[~nas_son],sites_sulphate_son_c[~nas_son])

	nas_djf = np.logical_or(np.isnan(gc_data_sulphate_djf), np.isnan(sites_sulphate_djf_c))
	correlate_djf = stats.pearsonr(gc_data_sulphate_djf[~nas_djf],sites_sulphate_djf_c[~nas_djf])

	#print('Correlation = ',correlate_Annual)

	return GC_surface_sulfate_msp, sites_sulphate_msp, sites_lon, sites_lat





GC_surface_sulfate_msp, sites_sulphate_msp, sites_lon, sites_lat = sulphate()
GC_surface_nitrate_msp, sites_nitrate_msp = nitrate()
GC_surface_HNO3_msp, sites_HNO3_msp = HNO3()
GC_surface_ammonium_msp, sites_ammonium_msp = Ammonium()
GC_surface_ammonia_msp, sites_ammonia_msp = NH3()
print (len(sites_ammonia_msp), 'sites_ammonia_msp')
print (len(sites_ammonium_msp), 'sites_ammonium_msp')
print (len(sites_HNO3_msp), 'sites_HNO3_msp')
print (len(sites_nitrate_msp), 'sites_nitrate_msp')
print (len(sites_sulphate_msp), 'sites_sulphate_msp')

GC_dig_ratio = ((GC_surface_ammonia_msp + GC_surface_ammonium_msp) - 2*(GC_surface_sulfate_msp))/(GC_surface_HNO3_msp+GC_surface_nitrate_msp)
#print (GC_dig_ratio)
DEFRA_dig_ratio = ((sites_ammonia_msp + sites_ammonium_msp) - 2*(sites_sulphate_msp))/(sites_HNO3_msp+sites_nitrate_msp)
#print (DEFRA_dig_ratio)
# plotting spatial map model and DEFRA network 
os.chdir('/home/a/ap744/scratch_alok/shapefiles/GBP_shapefile')
Europe_shape = r'GBR_adm1.shp'
Europe_map = ShapelyFeature(Reader(Europe_shape).geometries(),
                               ccrs.PlateCarree(), edgecolor='black',facecolor='none')
print ('Shapefile_read')

fig2 = plt.figure(facecolor='White',figsize=[11,11]);pad= 1.1;
ax = plt.subplot(232);
#plt.title(title_list1, fontsize = 30, y=1)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.add_feature(Europe_map)
political = cartopy.feature.NaturalEarthFeature(
	category='cultural',
	name='admin_0_boundary_lines_land',
	scale='10m',
	facecolor='none')

states_provinces = cartopy.feature.NaturalEarthFeature(
	category='cultural',
	name='admin_0_boundary_lines_map_units',
	scale='10m',
	facecolor='none')

ax.add_feature(political, edgecolor='black')
ax.add_feature(states_provinces, edgecolor='black')
ax.coastlines(resolution='10m')

ax.set_extent([-9, 3, 49, 61], crs=ccrs.PlateCarree()) # [lonW,lonE,latS,latN]
GC_dig_ratio.plot(ax=ax,cmap=cmap,vmin = 0,vmax =2,
								cbar_kwargs={'shrink': 0.6, 
											'pad' : 0.05,
											'label': 'Diagnostic Ratio (Mar-Sep)',
											'orientation':'horizontal'})
											
ax.scatter(x=sites_lon, y=sites_lat,c=DEFRA_dig_ratio,
		facecolors='none',edgecolors='black',linewidths=5,s = 100)
ax.scatter(x=sites_lon, y=sites_lat,c=DEFRA_dig_ratio,
		cmap=cmap,s = 100,vmin = 0,vmax = 3)
		
ax.set_title('Diagnostic Ratio (mar-sep)',fontsize=15)
PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes



		
#colorbar = plt.colorbar(PCM, ax=ax,label='GEOS-Chem & DEFRA ammonia ($\mu$g m$^{-3}$)',
#                        orientation='horizontal',shrink=0.5,pad=0.05)
#colorbar.ax.tick_params(labelsize=15) 
#colorbar.ax.xaxis.label.set_size(15)
#plt.savefig('/scratch/uptrop/ap744/python_work/'+Today_date+'DignosticRatio_Mar-Sep_iccw.png',bbox_inches='tight')
fig2.savefig('/scratch/uptrop/ap744/python_work/'+Today_date+'DignosticRatio_Mar-Sep_iccw.ps', format='ps')
plt.show()
