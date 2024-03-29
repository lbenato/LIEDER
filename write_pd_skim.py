import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
from samples import *
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
#gROOT.ProcessLine('.L Objects.h' )
#from ROOT import LeptonType, JetType, FatJetType, MEtType, MEtFullType, CandidateType, LorentzType

# storage folder of the original root files
folder = '/nfs/dust/cms/group/cms-llp/v7_calo/SkimMET/'
sgn = ['VBFH_M15_ctau100','VBFH_M20_ctau100','VBFH_M25_ctau100','VBFH_M15_ctau500','VBFH_M20_ctau500','VBFH_M25_ctau500','VBFH_M15_ctau1000','VBFH_M20_ctau1000','VBFH_M25_ctau1000','VBFH_M15_ctau2000','VBFH_M20_ctau2000','VBFH_M25_ctau2000','VBFH_M15_ctau5000','VBFH_M20_ctau5000','VBFH_M25_ctau5000','VBFH_M15_ctau10000','VBFH_M20_ctau10000','VBFH_M25_ctau10000']
bkg = ['ZJetsToNuNu','DYJetsToLL','WJetsToLNu','QCD','VV','TTbar','ST','DYJetsToQQ','WJetsToQQ']


# define your variables here
var_list = ['EventNumber','RunNumber','LumiNumber','EventWeight','isMC','isVBF','HT','MEt_pt','MEt_phi','MEt_sign','MinJetMetDPhi','nCHSJets','nElectrons','nMuons','nPhotons','nTaus','j0_pt','j1_pt','j0_nTrackConstituents','j1_nTrackConstituents','j0_nConstituents','j1_nConstituents','j0_nSelectedTracks','j1_nSelectedTracks','j0_nTracks3PixelHits','j1_nTracks3PixelHits','j0_nHadEFrac','j1_nHadEFrac','j0_cHadEFrac','j1_cHadEFrac']#,'c_nEvents']#,'is_signal']

variables = []


def write_h5(folder,output_folder,file_list,test_split,verbose=True):
    print("    Opening ", folder)
    print("\n")
    # loop over files
    for a in file_list:
        print(a)
        for i, ss in enumerate(samples[a]['files']):
            #read number of entries
            oldFile = TFile(folder+ss+'.root', "READ")
            counter = oldFile.Get("c_nEvents")#).GetBinContent(1)
            nevents_gen = counter.GetBinContent(1)
            print("  n events gen.: ", nevents_gen)
            oldTree = oldFile.Get("skim")
            nevents = oldTree.GetEntries()-1

            #initialize data frame
            df = pd.DataFrame(index = np.arange(nevents) ,columns=var_list)
            df = df.fillna(0)

            if verbose:
                print("\n")
                #print("   Initialized df for sample: ", file_name)
                print("   Initialized df for sample: ", ss)
                print("   Reading n. events: ", nevents)
                #print df #this prints empty df
                print("\n")

            if nevents<0:
                print("   Empty tree!!! ")
                continue

            # loop over variables
            for nr,variable in enumerate(var_list):
                b = rnp.root2array(folder+ss+'.root', treename='skim', branches=(variable), start=0, stop=nevents)
                df[variable] = b

            #add is_signal flag
            #df["is_signal"] = np.ones(nevents) if "VBFH_M" in fileNas[fi] else np.zeros(nevents)
            df["is_signal"] = np.ones(nevents) if "VBFH_" in ss else np.zeros(nevents)
            df["c_nEvents"] = np.ones(nevents) * nevents_gen

            if verbose:
                print(df)

            #split test and training samples
            #first shuffle
            df.sample(frac=1).reset_index(drop=True)

            #define train and test samples
            n_events = int(df.shape[0] * (1-test_split) )
            df_train = df.head(n_events)
            df_test  = df.tail(df.shape[0] - n_events)
            print("  -------------------   ")
            print("  Events for training: ", df_train.shape[0])
            print("  Events for testing: ", df_test.shape[0])

            #write h5
            df_train.to_hdf(output_folder+'/'+ss+'_train.h5', 'df', format='table')
            print("  "+output_folder+"/"+ss+"_train.h5 stored")
            df_test.to_hdf(output_folder+'/'+ss+'_test.h5', 'df', format='table')
            print("  "+output_folder+"/"+ss+"_test.h5 stored")
            print("  -------------------   ")


'''
def read_h5(folder,file_names):
    for fi, file_name in enumerate(file_names):
        #read hd5
        store = pd.HDFStore(folder+'numpy/'+fileNas[fi]+'.h5')
        df2 = store.select("df")
        print(df2)
'''

print("write")
write_h5(folder,"dataframes",sgn+bkg,test_split=0.2)

#print "read"
#read_h5(folder,file_names)
