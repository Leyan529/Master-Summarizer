PF_Tag_Structure = {
                            'JJ[O] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}}, 
                                                    {'TAG': {"IN": ['NN','NNS']}}] , [1], [0]), 
                            'JJ[O] NN[F]/NNS[F] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}},
                                                                {'TAG': {"IN": ['NN','NNS']}}, 
                                                                {'TAG': {"IN": ['NN','NNS']}}] , [1,2], [0]),
                            'JJ[O] JJ[O] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}}, 
                                                           {'TAG': {"IN": ['JJ']}}, 
                                                            {'TAG': {"IN": ['NN','NNS']}}] , [2], [0]), 
    
                            'NN[F]/NNS[F] IN[?] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['NN','NNS']}},
                                                                  {'TAG': {"IN": ['JJ']}}, 
                                                                {'TAG': {"IN": ['NN','NNS']}}] , [0,2], [0]), 
    
                            'NN[F]/NNS[F] IN[?] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['NN','NNS']}},
                                                                  {'TAG': {"IN": ['IN']}},
                                                                {'TAG': {"IN": ['DT']}}, 
                                                                  {'TAG': {"IN": ['NN','NNS']}}] , [0,3], [0]) 
}

# FO-Rule(POS)
# pattern_id : (structure , f_pos_list, o_pos_list)
POS_Tag_Structure = {
                            # high quality image
                            'JJ[O] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}}, 
                                                    {'TAG': {"IN": ['NN','NNS']}}] , [1], [0]), 
                            # amazing battery life
                            'JJ[O] NN[F]/NNS[F] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}},
                                                                {'TAG': {"IN": ['NN','NNS']}}, 
                                                                {'TAG': {"IN": ['NN','NNS']}}] , [1,2], [0]),

                            # amazed with the image
                            'JJ[O] IN DT NN[F]': (  [{'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['IN']}},
                                                {'TAG': 'DT'},
                                                {'TAG': {"IN": ['NN']}}] , [3], [0]),
                            # Good and excellent camera .
                            'JJ[O] CC JJ[O] NN[F] ': (  [{'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['CC']}},
                                                {'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['NN']}}] , [3], [0,2]),
                            # Performance is excellent .                     
                            'NN[F] VB/VBZ JJ[O]': (   [{'TAG': {"IN": ['NN']}},
                                                {'TAG': {"IN": ['VB','VBZ']}},
                                                {'TAG': {"IN": ['JJ']}}] , [0], [2]),
                            # Keanu performs well .                     
                            'NN/NNP[F] NN/VBZ JJ/RB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['NN','VBZ']}},
                                                {'TAG': {"IN": ['JJ','RB']}}] , [0], [2]),
                            # Keanu performs very well.                     
                            'NN/NNP[F] VB/VBZ RB RB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['VB','VBZ']}},
                                                {'TAG': {"IN": ['RB']}},
                                                {'TAG': {"IN": ['RB']}}] , [0], [3]),
                            # Siri can sometimes help .                     
                            'NN/NNP[F] MD RB VB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['MD']}},
                                                {'TAG': {"IN": ['RB']}},
                                                {'TAG': {"IN": ['VB']}}] , [0], [3]),
                            # I love this phone .                      
                            'VB/VBP[O] DT NN[F]': (   [{'TAG': {"IN": ['VB','VBP']}},
                                                {'TAG': {"IN": ['DT']}},
                                                {'TAG': {"IN": ['NN']}}] , [2], [0]),
                            # Camera is worse than N81 .                     
                            'NN[F] VBZ JJR[O] IN NN/NNP': (   [{'TAG': {"IN": ['NN']}},
                                                {'TAG': {"IN": ['VBZ']}},
                                                {'TAG': {"IN": ['JJR']}},
                                                {'TAG': {"IN": ['IN']}},
                                                {'TAG': {"IN": ['NN','NNP']}}] , [0], [2]),

                            # Support Bluetooth                     
                            'VB[O] NN/NNP[F]': (   [{'TAG': {"IN": ['VB']}},
                                                {'TAG': {"IN": ['NN','NNP']}}] , [1], [0]),

                            # --------------------------------------------------------------

                            # very excellent volume .                     
                            'RB/RBR/RBS JJ[O]  NN/NNS[F]': ( [{'TAG': {"IN": ['RB','RBR','RBS']}},
                                                        {'TAG': {"IN": ['JJ']}},
                                                        {'TAG': {"IN": ['NN',"NNS"]}}
                                                        ] , [2], [1]),

                            # # honestly recommend .                     
                            # 'RB/RBR/RBS JJ[O]  NN/NNS[F]': ( [{'TAG': {"IN": ['RB','RBR','RBS']}},
                            #                                 {'TAG': {"IN": ['VBN','VBD','VB']}}
                            #                                 ] , [2], [1]),

                            # look excellent .                     
                            'VB[O] JJ[F]': ( [{'TAG': {"IN": ['VB','VBN','VBD']}},
                                            {'TAG': {"IN": ['JJ']}} 
                                            ] , [0], [1]),
                            # exhausted very quickly .                     
                            'VBN/VBD/VB[F] RB/RBR/RBS RB/RBR/RBS[O]': ( [{'TAG': {"IN": ['VBN','VBD','VB']}},
                                            {'TAG': {"IN": ['RB','RBR','RBS']}},
                                            {'TAG': {"IN": ['RB','RBR','RBS']}}
                                            ] , [0], [2]),

                            # our rule 
                            # image quality high
                            'NN[F] NN[F] JJ[O]': ( [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}, 
                            {'TAG': {"IN": ['JJ']}}] , [0,1], [2])
}