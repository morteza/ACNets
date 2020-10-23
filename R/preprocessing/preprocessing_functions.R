

tidy_event_data <- function(D){
  
  # add block number
  Block_start <- D %>% 
    filter(str_detect(type, "start block")) %>% 
    mutate(block = str_remove_all(type, "start block ")) %>% 
    rename(start_index = event_index) %>% 
    select(block, start_index)
  
  Block_end <- D %>% 
    filter(str_detect(type, "end block")) %>% 
    mutate(block = str_remove_all(type, "end block ")) %>% 
    rename(end_index = event_index) %>% 
    select(block, end_index)
  
  # sometimes block end is missing:
  if (length(Block_end$block) != 8){
    if (D$participant_id[1] == "NVGP17" & D$session[1] == 1){
      Block_end <- rbind(Block_end, data.frame(block = "8", end_index = max(D$event_index)))
    }
  }
  
  Block_index <- Block_start %>% 
    left_join(Block_end, by = "block")
  
  
  D$block <- NA
  for (i in 1:length(Block_index$block)){
    D$block[Block_index$start_index[i]:Block_index$end_index[i]] <- Block_index$block[i]
  }
  
  
  # ---- add trial number ----
  n <- length(D$event_index)
  trial_index <- 0
  D$trial_index <- NA
  # browser()
  for (i in 1:n){
    if (str_detect(D$type[i], fixed("cue"))){
      # check that there's a stimulus in one of the subseqeunt events
      if (i < (n-5)){
        if (any(str_detect(D$type[(i+1):(i+5)], fixed("target")))){
          trial_index <- trial_index + 1
        }
      }

    }
    D$trial_index[i] <- trial_index
  }
  # there's an issue here because trial_index are not aligned with block_index
  
  # decompose type into type_key and type_value
  D <- D %>% 
    mutate(
      type_key = ifelse(str_detect(type, fixed("start session ")), "session_start",type),
      type_key = ifelse(str_detect(type, fixed("trigger")), "trigger",type_key),
      type_key = ifelse(str_detect(type, fixed("start block ")), "block_start",type_key),
      type_key = ifelse(str_detect(type, fixed("end block ")), "block_end",type_key),
      type_key = ifelse(str_detect(type, fixed("target")), "stimulus",type_key),
      type_key = ifelse(str_detect(type, fixed("cue")), "cue",type_key),
      type_key = ifelse(str_detect(type, fixed("response")), "response",type_key),
      type_key = ifelse(str_detect(type, fixed("missing arm")), "fixation",type_key)
    ) %>% mutate(
      type_value = ifelse(str_detect(type, fixed("start session ")), "session_start",type),
      type_value = ifelse(str_detect(type, fixed("trigger")), NA,type_value),
      type_value = ifelse(str_detect(type, fixed("block")), block,type_value),
      type_value = ifelse(str_detect(type, fixed("noise")), "noise",type_value),
      type_value = ifelse(str_detect(type, fixed("up")), "up",type_value),
      type_value = ifelse(str_detect(type, fixed("down")), "down",type_value),
      type_value = ifelse(str_detect(type, fixed("left")), "left",type_value),
      type_value = ifelse(str_detect(type, fixed("right")), "right",type_value)
    )
  
  
  
  # reorder columnes
  D <- D %>% select(c("participant_id", "group", "session",
                      "block", "trial_index", "event_index",   
                      "type_key", "type_value", "blockTime",  "realTime")) %>% 
    rename(session_index = session,
           block_index = block,
           event_type = type_key,
           event_value = type_value,
           timestamp_in_block = blockTime,
           timestamp_in_session = realTime)
  
  D
}



extract_trial_data_from_event_data <- function(D){
  
  
  # if there's a response check if it appeared after a stimulus or a cue;
  # if it's after a stimulus, compute RT; if it's after a cue express as negative;
  # if there's a stimulus which is not followed by an RT tag that trial as being a 
  # no-go trial
  # add trial number
  n <- length(D$event_index)
  D$rt <-NA
  D$SOA <- NA
  
  for (i in 1:n){
    # compute response times
    if (D$event_type[i] == "response"){
      response_timestamp <- D$timestamp_in_session[i]
      
      if (D$event_type[i-1] == "stimulus"){
        stimulus_timestamp <- D$timestamp_in_session[i-1]
      }
      
      if (D$event_type[i-1] == "fixation"){
        if (D$event_type[i-2] == "stimulus"){
          stimulus_timestamp <- D$timestamp_in_session[i-2]
        }
        if (D$event_type[i-2] == "fixation"){
          if (D$event_type[i-3] == "stimulus"){
            stimulus_timestamp <- D$timestamp_in_session[i-3]
          }
        }
        
        
      }
      
      if (D$event_type[i+1] == "stimulus"){
        stimulus_timestamp <- D$timestamp_in_session[i+1]
      }
      
      D$rt[i] <- response_timestamp - stimulus_timestamp
    }
    
    # extract SOA
    cue_timestamp <- NA # for some reason, cue_timestamp is missing in "VGP10_ses-02"
    if (D$event_type[i] == "stimulus"){
      stimulus_timestamp <- D$timestamp_in_session[i]
      if (D$event_type[i-1] == "cue"){
        cue_timestamp <- D$timestamp_in_session[i-1]
      }
      if (D$event_type[i-1] == "fixation"){
        if (D$event_type[i-2] == "cue"){
          cue_timestamp <- D$timestamp_in_session[i-2]
        }
        if (D$event_type[i-3] == "fixation"){
          if (D$event_type[i-3] == "cue"){
            cue_timestamp <- D$timestamp_in_session[i-3]
          }
          
          if (D$event_type[i-4] == "fixation"){
            if (D$event_type[i-4] == "cue"){
              cue_timestamp <- D$timestamp_in_session[i-4]
            }
        }
          
        }
      }
      
      
      D$SOA[i] <- stimulus_timestamp - cue_timestamp
    }
  }
  
  
  # ---- create "trial" table ----
  trial_n <- max(D$trial_index)
  Trial <- data.frame()
  
  for (current_trial in 1:trial_n){
    TMP <- filter(D, trial_index == current_trial)
    trial_index <- current_trial
    cue <- TMP$event_value[TMP$event_type == "cue"]
    stimulus <- TMP$event_value[TMP$event_type == "stimulus"]
    soa <- TMP$SOA[TMP$event_type == "stimulus"]
    fixation_n <- sum(TMP$event_type == "fixation")
    response_n <- length(TMP$event_value[TMP$event_type == "response"])
    
    response <- NA
    rt <- NA
    if (response_n >= 1){
      # if is one or more responses; take the last one.
      # maybe first that has postivie rt?
      response <- TMP$event_value[TMP$event_type == "response"][response_n]
      rt <- TMP$rt[TMP$event_type == "response"][response_n]
    }
    
    Trial <- rbind(Trial, data.frame(trial_index, cue, stimulus, soa, fixation_n,response_n, response, rt))
  }
  
  # evaluate correctnes of the response:
  Trial$stimulus <- as.character(Trial$stimulus)
  Trial$response <- as.character(Trial$response)
  Trial$correct <- Trial$stimulus == Trial$response
  Trial$correct[Trial$stimulus == "noise" & is.na(Trial$response)] <- TRUE
  Trial$correct[Trial$stimulus != "noise" & is.na(Trial$response)] <- FALSE
  
  # there's an issue where the same trial can belong to two differeng blocks
  Trial$block_index <- ceiling(Trial$trial_index / 20)
  
  TMP <- D %>% 
    select(participant_id:trial_index) %>% 
    select(-block_index) %>% 
    filter(trial_index > 0) %>% 
    distinct() %>% 
    left_join(Trial, by = "trial_index") %>% 
    select(c("participant_id", "group", "session_index",
             "block_index", "trial_index", 
             "soa", "fixation_n", 
             "cue", "stimulus", 
             "response_n", "response", "rt", "correct"))       
  TMP
}



# there are some issues with the data:
# there can be multiple "missing arm" events per trial
# there can be multiple key presses during a trial (e.g. anticipations)

