from behavioral_clustering import BehavioralClustering



def get_example_pipeline():
    filenames = [
        "26148_020520_bank0_s1_light.pkl",
        "26148_030520_bank0_s2_light.pkl",
    ]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")
    
    bc.load_relevant_features(filenames)
    bc.remove_nan()
    bc.detrend()

    return bc


def main():
    filenames = [
        "26148_020520_bank0_s1_light.pkl",
        "26148_030520_bank0_s2_light.pkl",
    ]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")
    
    # bc.pickle_relevant_features(filenames)
    bc.load_relevant_features(filenames)
    bc.remove_nan()
    bc.detrend()





if __name__ == "__main__":
    main()

