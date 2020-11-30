% Use relative path starting from the project directory.
in_dir = 'data/julia2018_raw/Behavioral_Data/';
out_dir = 'data/julia2018/sourcedata/raw_behavioral/';


mat_files = dir(fullfile(in_dir,'**/*.mat'));

if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end
    
for f = 1:length(mat_files)
  file = mat_files(f);
  full_path = fullfile(file.folder, file.name);
  disp(full_path);
  data = load(full_path);

  % `frame` embeds another table which might not be useful right now.
  trial_headers = fieldnames(rmfield(data.trial,'frame'));
  data.trials = data.trials(:,1:end-1);
  data.trial = cell2struct(data.trials, trial_headers, 2);
  
  writetable(struct2table(data.trial),strcat(out_dir, strrep(file.name,'.mat','_trials.csv')));
  writetable(struct2table(data.block),strcat(out_dir, strrep(file.name,'.mat','_blocks.csv')));
  writetable(struct2table(data.event),strcat(out_dir, strrep(file.name,'.mat','_events.csv')));

  if isfield(data,'p')
    writetable(struct2table(data.p),strcat(out_dir, strrep(file.name,'.mat','_parameters.csv')));
  end
end
