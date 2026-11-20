% Score the toolbox's spare full-resolution outputs (methods not in Table III).
root = 'F:\taskA\old\toolbox\Toolbox_psnr\Toolbox_psnr';
addpath(genpath(root)); cd(root);

sets = {
  'WV3', 11, 'test_wv3_OrigScale_multiExm1.h5', 'WV3_Full',  {'BDPN','MMNet','MSDCNN','MUCNN','PMAC'}
  'GF2', 10, 'test_gf2_OrigScale_multiExm1.h5', 'GF2_Full',  {'BDPN','MMNet','MSDCNN','MUCNN','PMACNet'}
  'QB',  10, 'test_qb_OrigScale_multiExm1.h5',  'QB_Full',   {'BDPN','MMNet','MSDCNN','MUCNN','PMACNet'}
};

for s = 1:size(sets,1)
    sensor = sets{s,1}; L = sets{s,2}; h5n = sets{s,3};
    sub = sets{s,4}; methods = sets{s,5};
    h5 = fullfile('F:\taskA\PanCollection\full_examples', h5n);
    ms = h5read(h5, '/ms'); lms = h5read(h5, '/lms'); pan = h5read(h5, '/pan');
    N = size(ms, 4);
    for k = 1:numel(methods)
        d = fullfile('F:\taskA\old\toolbox\Toolbox_psnr\Toolbox_psnr\2_DL_Result', sub, methods{k}, 'results');
        dl_ = 0; ds_ = 0; q_ = 0; n = 0;
        for i = 0:N-1
            f = fullfile(d, sprintf('output_mulExm_%d.mat', i));
            if ~exist(f, 'file'), continue; end
            S = load(f); F = double(S.sr);
            I_MS_LR = double(permute(ms(:,:,:,i+1), [2 1 3]));
            I_MS    = double(permute(lms(:,:,:,i+1), [2 1 3]));
            I_PAN   = double(squeeze(permute(pan(:,:,1,i+1), [2 1 3])));
            [dl, ds, q] = indexes_evaluation_FS(F, I_MS_LR, I_PAN, L, 0, I_MS, sensor, 4, 32, 0);
            dl_ = dl_ + dl; ds_ = ds_ + ds; q_ = q_ + q; n = n + 1;
        end
        if n == 0
            fprintf('EXTRA %-4s %-10s  no outputs\n', sensor, methods{k});
        else
            fprintf('EXTRA %-4s %-10s  Dl %.4f  Ds %.4f  HQNR %.4f  (n=%d)\n', ...
                    sensor, methods{k}, dl_/n, ds_/n, q_/n, n);
        end
    end
end
fprintf('EXTRA DONE\n');
