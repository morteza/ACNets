# The Behavioral data from the Foecker et al. (2018) study is stored as mat files (Matlab).
# This script loops over those mat files, extract data from them and saves them as distinct tidy tables in csv format.

library(tidyverse)
library(R.matlab)


# list of participants
input_data_dir <- "../../data/Behavioral_Data/"
output_data_dir <- "../../data/Behavioral_Data_csv/"

dir.create(output_data_dir)

participants <- list.dirs(input_data_dir, full.names = FALSE, recursive = FALSE)
column_names <- c("type", "gabor", "cue", "deviant", "SOA", "ITI", "trialTime", "missingArms")




for (participant_id in participants){
  for (session in 1:2){
    
    data_file <-paste0(input_data_dir, participant_id,"/A", session, "/", participant_id, "_A", session, ".mat") 
    participant_group <- ifelse(str_detect(participant_id, fixed("NVGP"), ), "NVGP", "AVGP")
    
    
    # check if data exists
    if (!file.exists(data_file)){
      
      print(paste0("Missing data file: ", data_file))
      
    } else {
      
      
      M <- readMat(data_file)
      
      
   
      
      Trial <- data.frame()
      Frame <- data.frame()
      trial_n <- length(M$trial[1,1,])
      for (trial_index in 1:trial_n){
        
        # fill trial table
        tmp <- M$trial[,1,trial_index]
        tmp <- data.frame(tmp[column_names])
        tmp$participant_id <- participant_id
        tmp$group <- participant_group
        tmp$session <- session
        tmp$trial_index <- trial_index
        Trial <- rbind(Trial, tmp[c("participant_id", "group", "session", "trial_index", column_names)])
        
        # frame data:
        tmp <- M$trial[,1,trial_index]
        
        Frame_tmp <- as_tibble(tmp$frame[,,1])
        # Frame_tmp <- as.data.frame(t(as.data.frame(tmp$frame)))
        # frame_names <- names(Frame_tmp)
        frame_names <- c("time", "isMissing", "canMiss")
        names(Frame_tmp) <- frame_names
        Frame_tmp$participant_id <- participant_id
        Frame_tmp$group <- participant_group
        Frame_tmp$session <- session
        Frame_tmp$trial_index <- trial_index
        Frame_tmp$frame_index <- 1:length(length(tmp$frame[1,1,])) # number of frames in current trial
        Frame <- rbind(Frame, Frame_tmp[c("participant_id", "group", "session", "trial_index", "frame_index", frame_names)])
      }
      Frame <- as.data.frame(Frame)
      # row.names(Frame) <- NULL
      # rm(list = c("tmp", "Frame_tmp"))
      
      # some information is stored by block:
      B <- M$block
      block_n <- length(B[1,1,])
      Block <- data.frame()
      for (block_index in 1:block_n){
        
        # 
        
        armActual <- unlist(B[,,block_index]$armActual)
        duration <- unlist(B[,,block_index]$duration)
        armResponse <- unlist(B[,,block_index]$armResponse)
        
        armActual <- ifelse(is.null(armActual), NA, armActual)
        duration <- ifelse(is.null(duration), NA, duration)
        armResponse <- ifelse(is.null(armResponse), NA, armResponse)
        
        B_tmp <- tibble(armActual,duration,armResponse)
        
        
        # B_tmp <- as.data.frame(B[,,block_index])
        tmp_names <- names(B_tmp)
        
        B_tmp$participant_id <- participant_id
        B_tmp$group <- participant_group
        B_tmp$session <- session
        B_tmp$block_index <- block_index
        
        Block <- rbind(Block, B_tmp[c("participant_id", "group", "session", "block_index", tmp_names)])
      }
      
      
      # M$blocks
      # list of 24?? contains the columns of block in a long vector without variable name => useless
      
      
      
      # not clear what difference is between event and events;
      E <- M$event
      event_n <- length(E[1,1,])
      Event <- data.frame()
      for (event_index in 1:event_n){
        E_tmp <- as.data.frame(E[,,event_index])
        tmp_names <- names(E_tmp)
        
        E_tmp$participant_id <- participant_id
        E_tmp$group <- participant_group
        E_tmp$session <- session
        E_tmp$event_index <- event_index
        
        Event <- rbind(Event, E_tmp[c("participant_id", "group", "session", "event_index", tmp_names)])
      }
      
      # save data as csv files
      write_csv(x = Trial, path = paste0(output_data_dir, participant_id, "_ses-0", session, "_trial.csv"))
      write_csv(x = Frame, path = paste0(output_data_dir, participant_id, "_ses-0", session, "_frame.csv"))
      write_csv(x = Block, path = paste0(output_data_dir, participant_id, "_ses-0", session, "_block.csv"))
      write_csv(x = Event, path = paste0(output_data_dir, participant_id, "_ses-0", session, "_event.csv"))
      
      # sometimes there are user parameters:
      if (!is.null(M$p)){
        Parameters <- as.data.frame(M$p[,,1])
        write_csv(x = Parameters, path = paste0(output_data_dir, participant_id, "_ses-0", session, "_parameter.csv"))
      }
      
      
      
    }
  }
}
