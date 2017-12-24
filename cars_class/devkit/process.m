load('cars_train_annos.mat');

fid=fopen('anno.txt', 'w');

for i = 1:8144
    if (annotations(i).class == 1) || (annotations(i).class == 2)
        fprintf(fid, '%d %s\n', [annotations(i).class, annotations(i).fname]);
    end
end

fclose(fid)
