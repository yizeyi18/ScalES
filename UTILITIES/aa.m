if(0)
  binstr = 'pseudopot';
  fid = fopen(binstr,'r');
  string = {'vector', {'NumTns',{'DblNumTns'}}};
  val = deserialize(fid, string);
  fclose(fid);
  
  for g=1:numel(val)
    tmp = val{g};
    for h=1:size(tmp,1)
      ee = tmp{h};
      for i=1:size(tmp,3)
        imagesc(ee(:,:,i)); colorbar; pause(0.5);
      end
    end
  end
end

if(1)
  binstr = 'now';
  fid = fopen(binstr,'r');
  string = {'NumTns', {'vector',{'DblNumTns'}}};
  val = deserialize(fid, string);
  fclose(fid);
  
  for g=1:numel(val)
    tmp = val{g};
    for h=1:size(tmp,1)
      ee = tmp{h};
      % for i=1:size(tmp,3)
        % imagesc(ee(:,:,i)); colorbar; pause(0.5);
      % end
    end
  end
end

imagesc(squeeze(val{1}{1}(20,:,:))); colorbar;
maxval = max(max(squeeze(val{1}{1}(20,:,:))))

