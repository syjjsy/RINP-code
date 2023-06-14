def corrf(input):
    ft1=torch.fft.fft2(input)
    fpr=ft1.mul(torch.conj(ft1))
    pr=torch.fft.ifft2(fpr)
                # print(pr)
    corr=torch.abs(torch.fft.fftshift(pr))
    maxcorr=torch.max(corr)
    ninbg=(torch.sqrt(8*maxcorr +1)-1)/2
    cinbg=maxcorr*ninbg/(ninbg+1)
    outcorr=corr-cinbg
    maxout=torch.max(outcorr)
    minout=torch.min(outcorr)
    detout=1/(maxout-minout)
    outcorr1=(outcorr-minout)*detout
    return outcorr1