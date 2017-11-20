2. Add entropy term to encourage exploration
3. GAE
4. Distributional 
5. Other environments 
6. Bigger -> SLower nets
7. The exploration noise causes NAN gradients, thus NAN outputs
8. Need experience replay because it's OBVIOUSLY forgetting stuff from the past. 
9. Use OpenAI examples


1. Combine 2 nets into one -> Works -> Learns a bit slower I think
2. Tuned hyper-parameters, specifically the size of roll-outs, number of updates and batch size
3. Next step -> Try GAE estimation
4. After -> Train in distributed setting with harder environments
5. Compare to OpenAI baseline

6. Incorporate into StarCraft