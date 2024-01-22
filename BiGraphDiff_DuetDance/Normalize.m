clear
data_path = '\Skeletons_full_seq';
save_path = '\skeletons_normalized_inter_short';
mean_path = '\MEANS_inter_short';
norm_path = '\NORMS_inter_short';
center_path = '\CENTERS_inter_short';


max_frames = 300;

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)%%test 4->1 pour vrai
    data_subfolder=dir([data_path '\' data_folder(i).name]);
    data_subfolder=data_subfolder(3:end);
    
    mkdir([save_path '\' data_folder(i).name]);
    mkdir([mean_path '\' data_folder(i).name]);
    mkdir([norm_path '\' data_folder(i).name]);
    mkdir([center_path '\' data_folder(i).name]);
    fprintf("%d\n",i);
    nb_fold= 1;
    for j=1:length(data_subfolder)
        skel_A = load([data_path '\' data_folder(i).name '\' data_subfolder(j).name '\skeleton_A.mat']);
        skel_B = load([data_path '\' data_folder(i).name '\' data_subfolder(j).name '\skeleton_B.mat']);
        skel_A = skel_A.skeletons;
        skel_B = skel_B.skeletons;
        nb_frames = size(skel_A,2);
        if nb_frames<=max_frames
            SKELA=[];
            SKELB= [];
            mkdir([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
            mkdir([mean_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
            mkdir([norm_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
            mkdir([center_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
            for k=1:nb_frames
                pts_A = [skel_A(1:15,k),skel_A(16:30,k),skel_A(31:45,k)];
                pts_B =[skel_B(1:15,k),skel_B(16:30,k),skel_B(31:45,k)];
                pts_inter = [pts_A;pts_B];
                if k==1
                    m1 = mean(pts_inter(:,1));
                    m2 = mean(pts_inter(:,2));
                    m3 = mean(pts_inter(:,3));
                    pts1 = pts_inter - [m1,m2,m3];
                    normFro = norm( pts1 ,'fro');
                    pts1 = pts1 / normFro;
                    xc = (pts1(14,1)+pts1(19,1))/2;
                    yc = (pts1(14,2)+pts1(19,2))/2;
                    zc = (pts1(14,3)+pts1(19,3))/2;
                    centering = [xc yc zc];
                    norms = normFro;
                    means = [m1, m2, m3];
                    Mean = [mean_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\mean.mat' ];
                    Norm = [norm_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\norm.mat' ];
                    Center = [center_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\center.mat' ];
                    save(Mean,'means');
                    save(Norm,'norms');
                    save(Center,'centering');
                    
                end
                pts1 = pts_A - [m1,m2,m3];
                pts1 = pts1 / normFro;
                x = pts1(:,1)- xc;
                y = pts1(:,2)- yc;
                z = pts1(:,3)- zc;
                ptsc_A = [x; y; z];
                SKELA = [SKELA,ptsc_A];
                pts1 = pts_B - [m1,m2,m3];
                pts1 = pts1 / normFro;
                x = pts1(:,1)- xc;
                y = pts1(:,2)- yc;
                z = pts1(:,3)- zc;
                ptsc_B = [x; y; z];
                SKELB = [SKELB,ptsc_B];
            end
            skel=SKELA;
            save([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\skeleton_A.mat'],'skel');
            skel=SKELB;
            save([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\skeleton_B.mat'],'skel');
            nb_fold =nb_fold+1;
        else
            nb_seq = floor(nb_frames/max_frames);
            if  rem(nb_frames,max_frames)>50
                nb_seq = nb_seq+1;
            end
            for s=1:nb_seq
                start=1+(s-1)*max_frames;
                finish=min(s*max_frames,nb_frames);
                SKELA=[];
                SKELB= [];
                mkdir([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
                mkdir([mean_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
                mkdir([norm_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
                mkdir([center_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d')]);
                for k=start:finish
                    pts_A = [skel_A(1:15,k),skel_A(16:30,k),skel_A(31:45,k)];
                    pts_B =[skel_B(1:15,k),skel_B(16:30,k),skel_B(31:45,k)];
                    pts_inter = [pts_A;pts_B];
                    if k==start
                        m1 = mean(pts_inter(:,1));
                        m2 = mean(pts_inter(:,2));
                        m3 = mean(pts_inter(:,3));
                        pts1 = pts_inter - [m1,m2,m3];
                        normFro = norm( pts1 ,'fro');
                        pts1 = pts1 / normFro;
                        xc = (pts1(14,1)+pts1(19,1))/2;
                        yc = (pts1(14,2)+pts1(19,2))/2;
                        zc = (pts1(14,3)+pts1(19,3))/2;
                        centering = [xc yc zc];
                        norms = normFro;
                        means = [m1, m2, m3];
                        Mean = [mean_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\mean.mat' ];
                        Norm = [norm_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\norm.mat' ];
                        Center = [center_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\center.mat' ];
                        save(Mean,'means');
                        save(Norm,'norms');
                        save(Center,'centering');
                    end
                    pts1 = pts_A - [m1,m2,m3];
                    pts1 = pts1 / normFro;
                    x = pts1(:,1)- xc;
                    y = pts1(:,2)- yc;
                    z = pts1(:,3)- zc;
                    ptsc_A = [x; y; z];
                    SKELA = [SKELA,ptsc_A];
                    pts1 = pts_B - [m1,m2,m3];
                    pts1 = pts1 / normFro;
                    x = pts1(:,1)- xc;
                    y = pts1(:,2)- yc;
                    z = pts1(:,3)- zc;
                    ptsc_B = [x; y; z];
                    SKELB = [SKELB,ptsc_B];
                end
                skel=SKELA;
                save([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\skeleton_A.mat'],'skel');
                skel=SKELB;
                save([save_path '\' data_folder(i).name '\' num2str(nb_fold,'%04d') '\skeleton_B.mat'],'skel');
                nb_fold =nb_fold+1;
            end
            
        end
    end
end




