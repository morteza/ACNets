# The Behavioral data from the Foecker et al. (2018) study is stored as mat files (Matlab).
# This script loops over those mat files, extract data from them and saves them as distinct tidy tables in csv format.
rm(list = ls())
library(tidyverse)
source("../preprocessing_behavioral_data/preprocessing_functions.R")

# list of participants
input_data_dir <- "../../data/Behavioral_Data_csv/"
output_data_dir <- "../../data/Behavioral_Tidy/"

if (!dir.exists(output_data_dir)){
  dir.create(output_data_dir)
}


file_names <- str_remove_all(dir(input_data_dir), ".csv")
file_names <- data.frame(file_names) %>% 
  separate(file_names, 
           into = c("participant_id", "session", "data_type"), 
           sep = "_") %>% 
  filter(data_type == "trial") %>% 
  select(-data_type) %>% 
  distinct() %>% 
  arrange(session, participant_id) %>% 
  unite("file_names", participant_id, session, sep = "_") %>% 
  pull(file_names)


for (file_name in file_names){
  
  Event <- read_csv(paste0(input_data_dir,file_name, "_event.csv"))
  Trial <- read_csv(paste0(input_data_dir, file_name, "_trial.csv"))
                    
  Event_tidy <- tidy_event_data(Event)
  Trial_tidy <- extract_trial_data_from_event_data(Event_tidy)
  
  
  Trial <- Trial %>% 
    rename(cue_type = cue) %>% 
    left_join(Trial_tidy)
  
  
  # save data as csv files
  write_csv(x = Event_tidy, path = paste0(output_data_dir, file_name, "_event.csv"))
  write_csv(x = Trial, path = paste0(output_data_dir, file_name, "_trial.csv"))
  
  
}

