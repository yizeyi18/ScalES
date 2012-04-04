if(1)
                    tmp = Ct*Gt;
                    tmp = tmp(:,1);
                    %accumulate
                    now = cell(3,3,3);
                    for g1=1:3
                        for g2=1:3
                            for g3=1:3
                                basis = basis_aux{g1,g2,g3};
                                nbasis = size(basis,1);
                                cur = zeros(Nlbl1,Nlbl2,Nlbl3);
                                ind = INDEXt{g1,g2,g3};
                                for a=1:nbasis
                                    cur = cur + basis{a} * tmp(ind(a));
                                end
                                now{g1,g2,g3} = cur;
                            end
                        end
                    end
                    tx1 = [x1-h1; x1; x1+h1];
                    tx2 = [x2-h2; x2; x2+h2];
                    tx3 = [x3-h3; x3; x3+h3];
                    all = zeros(3*Nlbl1, 3*Nlbl2, 3*Nlbl3);
                    for g1=1:3
                        for g2=1:3
                            for g3=1:3
                                all([1:Nlbl1]+(g1-1)*Nlbl1,[1:Nlbl2]+(g2-1)*Nlbl2,[1:Nlbl3]+(g3-1)*Nlbl3) = now{g1,g2,g3};
                            end
                        end
                    end
                    figure(1); imagesc(tx1,tx2, squeeze(all(:,:,round((end+1)/2)))); colorbar;
                    figure(2); imagesc(tx2,tx3, squeeze(all(round((end+1)/2),:,:))); colorbar;
                    figure(3); imagesc(tx3,tx1, squeeze(all(:,round((end+1)/2),:))); colorbar;
                end

if(0)
        tmp = C*Vc(:,4);
        %accumulate
        now = cell(N1,N2,N3);
        for g1=1:N1
            for g2=1:N2
                for g3=1:N3
                    basis = basis_all{g1,g2,g3};
                    nbasis = size(basis,1);
                    cur = zeros(Nlbl1,Nlbl2,Nlbl3);
                    ind = INDEX{g1,g2,g3};
                    for a=1:nbasis
                        cur = cur + basis{a} * tmp(ind(a));
                    end
                    now{g1,g2,g3} = cur;
                end
            end
        end
        tx1 = [];        for g1=1:N1            tx1 = [tx1; x1+(g1-1/2)*h1];        end
        tx2 = [];        for g2=1:N2            tx2 = [tx2; x2+(g2-1/2)*h2];        end
        tx3 = [];        for g3=1:N3            tx3 = [tx3; x3+(g3-1/2)*h3];        end
        all = zeros(N1*Nlbl1, N2*Nlbl2, N3*Nlbl3);
        for g1=1:N1
            for g2=1:N2
                for g3=1:N3
                    all([1:Nlbl1]+(g1-1)*Nlbl1,[1:Nlbl2]+(g2-1)*Nlbl2,[1:Nlbl3]+(g3-1)*Nlbl3) = now{g1,g2,g3};
                end
            end
        end
        figure(1); imagesc(tx1,tx2, squeeze(all(:,:,round((end+1)/2)))); colorbar;
        figure(2); imagesc(tx2,tx3, squeeze(all(round((end+1)/2),:,:))); colorbar;
        figure(3); imagesc(tx3,tx1, squeeze(all(:,round((end+1)/2),:))); colorbar;
        
        %figure(2); plot(tx1, all(:,round((end+1)/2),round((end+1)/2))); colorbar;
        %figure(3); plot(tx3, all(round((end+1)/2),:,round((end+1)/2))); colorbar;
    end