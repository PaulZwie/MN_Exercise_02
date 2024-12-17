function [STA_mean]=SpikeTriggeredAveraging(SIG,MUPulses,STA_window,fsamp)
% STA_mean= cell array of all motor units with MUAP shapes of each channel
% SIG = EMG Signal as a cell array aranged like the electrode grid shape
% MUPulses = Samples of MU discharges of all MUs
% STA_window = Window length STA should apply around spike timings [Seconds]
% fsamp = Sampling frequency of EMG signal acquisition

STA_window = round((STA_window/2)*fsamp);

for N = 1:size(MUPulses,2)
  
    for row = 1:size(SIG,1)
        for col = 1:size(SIG,2)
            if ~isempty(SIG{row,col})
                for spks = 1:size(MUPulses{N},2)
                    if MUPulses{N}(spks)+STA_window < length(SIG{row,col}) && MUPulses{N}(spks)-STA_window>=1
                        temp_STA(spks,:) = SIG{row,col}(MUPulses{N}(spks)-STA_window:MUPulses{N}(spks)+STA_window);
                    end
                end


                STA_mean{N}{row,col} =mean(temp_STA,'omitnan');
        
%                 STA_P2P{N}(row,col) = peak2peak(nanmean(temp_STA,1));
          
            end
        end
    end
end