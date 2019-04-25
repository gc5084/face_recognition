function  plotCumsumPCA(explained)
figure();
hold on;
cs = cumsum(explained); % to specify range --> explained(1:40, :)
xlabel('Dimension');
ylabel('Eigenvalue');
title("Cumulative Variance Explained")
plot(cs)
hold off;
end

