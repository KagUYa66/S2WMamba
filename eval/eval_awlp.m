% Score the classical AWLP baseline on PanCollection, reduced and full resolution.
root = 'F:\taskA\old\toolbox\Toolbox_psnr\Toolbox_psnr';
addpath(genpath(root)); cd(root);

specs = {
  'WV3', 11, 'test_wv3'
  'GF2', 10, 'test_gf2'
  'QB',  10, 'test_qb'
};

for s = 1:size(specs,1)
    sensor = specs{s,1}; L = specs{s,2}; stem = specs{s,3};
    maxvalue = 2047;

    % ---- reduced resolution ----
    f = fullfile(root,'1_TestData','PanCollection',[stem '_multiExm1.h5']);
    gt  = permute(h5read(f,'/gt'),  [4 2 1 3]);
    lms = permute(h5read(f,'/lms'), [4 2 1 3]);
    pan = permute(h5read(f,'/pan'), [4 2 1 3]);
    N = size(gt,1);
    ps = zeros(1,N); sa = ps; er = ps; qq = ps;
    for i = 1:N
        F = AWLP(squeeze(lms(i,:,:,:)), squeeze(pan(i,:,:,1)), 4);
        [p,~,~,sam,erg,~,~,q2n,~] = indexes_evaluation(double(F), double(squeeze(gt(i,:,:,:))), 4, L, 32, 1, 30, 0, maxvalue);
        ps(i)=p; sa(i)=sam; er(i)=erg; qq(i)=q2n;
    end
    fprintf('AWLP %-4s RR  PSNR %.4f  SAM %.4f  ERGAS %.4f  Q %0.4f\n', ...
            sensor, mean(ps), mean(sa), mean(er), mean(qq));

    % ---- full resolution ----
    f2 = fullfile('F:\taskA\PanCollection\full_examples', [stem '_OrigScale_multiExm1.h5']);
    ms2  = h5read(f2,'/ms');
    lms2 = h5read(f2,'/lms');
    pan2 = h5read(f2,'/pan');
    M = size(ms2,4);
    dl_ = 0; ds_ = 0; hq_ = 0;
    for i = 1:M
        I_MS_LR = double(permute(ms2(:,:,:,i),  [2 1 3]));
        I_MS    = double(permute(lms2(:,:,:,i), [2 1 3]));
        I_PAN   = double(squeeze(permute(pan2(:,:,1,i), [2 1 3])));
        F = AWLP(I_MS, I_PAN, 4);
        [dl, ds, qv] = indexes_evaluation_FS(double(F), I_MS_LR, I_PAN, L, 0, I_MS, sensor, 4, 32, 0);
        dl_ = dl_ + dl; ds_ = ds_ + ds; hq_ = hq_ + qv;
    end
    fprintf('AWLP %-4s FR  Dl %.4f  Ds %.4f  HQNR %.4f\n', sensor, dl_/M, ds_/M, hq_/M);
end
fprintf('AWLP DONE\n');
