import xml.etree.ElementTree as ET

__all__ = ['get_capabilities']

def get_capabilities(layers):

    root_args = {'xmlns':"http://www.opengis.net/wms",
                 'xmlns:xlink':"http://www.w3.org/1999/xlink",
                 'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance", 
                 'xsi:schemaLocation': 'http://www.opengis.net/wms',
                 'version':'1.3.0', 'updateSequence':'1102'}
    root = ET.Element('WMS_Capabilities', **root_args)

    service = ET.SubElement(root, 'Service')

    ET.SubElement(service, 'Name').text = 'WMS'
    ET.SubElement(service, 'Title')
    ET.SubElement(service, 'Abstract')
    ET.SubElement(service, 'KeywordList')
    ET.SubElement(service, 'OnlineResource')
    
    contact_info = ET.SubElement(service, 'ContactInformation')

    contact_person = ET.SubElement(contact_info, 'ContactPersonPrimary')

    ET.SubElement(contact_person, 'ContactPerson').text = "Dave Sproson"
    ET.SubElement(contact_person, 'ContactOrganization').text = 'My Company'
    ET.SubElement(contact_person, 'ContactPosition').text = (
                        'My Position')

    contact_address = ET.SubElement(contact_info, 'ContactAddress')

    ET.SubElement(contact_address, 'AddressType')
    ET.SubElement(contact_address, 'Address')
    ET.SubElement(contact_address, 'City')
    ET.SubElement(contact_address, 'StateOrProvince')
    ET.SubElement(contact_address, 'PostCode')

    ET.SubElement(contact_info, 'ContactVoiceTelephone')
    ET.SubElement(contact_info, 'ContactFacsilileTelephone')
    ET.SubElement(contact_info, 'ElectronicEmailAddress').text = (
                        'Fake@email.com')

    ET.SubElement(service, 'Fees').text = 'none'
    ET.SubElement(service, 'AccessConstraints').text = 'none'

    capability = ET.SubElement(root, 'Capability')
    request = ET.SubElement(capability, 'Request')

    get_capabilities = ET.SubElement(request, 'GetCapabilities')
    ET.SubElement(get_capabilities, 'Format').text = 'text/xml'

    dcp_type = ET.SubElement(get_capabilities, 'DCPType')
    http = ET.SubElement(dcp_type, 'HTTP')
    get = ET.SubElement(http, 'Get')
    ET.SubElement(get, 'OnlineResource', **{
                    'xlink:type': 'simple', 
                    'xlink:href': 'http://fgwfcluster3:8888/?SERVICE=WMS&'})



    get_map = ET.SubElement(request, 'GetMap')
    ET.SubElement(get_map, 'Format').text = 'image/png'
    dcp_type = ET.SubElement(get_map, 'DCPType')
    http = ET.SubElement(dcp_type, 'HTTP')
    get = ET.SubElement(http, 'Get')
    ET.SubElement(get, 'OnlineResource', **{
                    'xlink:type': 'simple', 
                    'xlink:href': 'http://fgwfcluster3:8888/?SERVICE=WMS&'})


#    get_feature_info = ET.SubElement(request, 'GetFeatureInfo')

    exception = ET.SubElement(capability, 'Exception')
    ET.SubElement(exception, 'Format').text = 'BLANK'

    root_layer = ET.SubElement(capability, 'Layer')
    ET.SubElement(root_layer, 'Title')
    ET.SubElement(root_layer, 'Abstract')

    valid_projections = ['EPSG:4326', 'EPSG:3857']
    for p in valid_projections:
        ET.SubElement(root_layer, 'CRS').text = p

    bbox = ET.SubElement(root_layer, 'EX_GeographicBoundingBox')
    ET.SubElement(bbox, 'westBoundLongitude').text = '-180'
    ET.SubElement(bbox, 'eastBoundLongitude').text = '180'
    ET.SubElement(bbox, 'southBoundLatitude').text = '-90'
    ET.SubElement(bbox, 'northBoundLatitude').text = '90'
    
    ET.SubElement(root_layer, 'BoundingBox',
            CRS='CRS:84', 
            minx='-180', miny='-90',
            maxx='180', maxy='90')

    for _layer in layers:
        layer = ET.SubElement(root_layer, 'Layer')
        ET.SubElement(layer, 'Name').text = 'TestLayer'
        ET.SubElement(layer, 'Title').text = 'Test Layer'
        ET.SubElement(layer, 'Abstract').text = 'Testing WMS Server'

        keyword_list = ET.SubElement(layer, 'KeywordList')
        ET.SubElement(keyword_list, 'Keyword').text = 'WMS'
        ET.SubElement(keyword_list, 'Keyword').text = 'Python'

        ET.SubElement(layer, 'CRS').text = 'EPSG:4326'
        
        bbox = ET.SubElement(layer, 'EX_GeographicBoundingBox')
        ET.SubElement(bbox, 'westBoundLongitude').text = '{}'.format(_layer.bbox[0])
        ET.SubElement(bbox, 'eastBoundLongitude').text = '{}'.format(_layer.bbox[2])
        ET.SubElement(bbox, 'southBoundLatitude').text = '{}'.format(_layer.bbox[1])
        ET.SubElement(bbox, 'northBoundLatitude').text = '{}'.format(_layer.bbox[3])

        ET.SubElement(layer, 'BoundingBox',
                   CRS='CRS:84', 
                   minx='{}'.format(_layer.bbox[0]), 
                   miny='{}'.format(_layer.bbox[1]),
                   maxx='{}'.format(_layer.bbox[2]), 
                   maxy='{}'.format(_layer.bbox[3]))

        style = ET.SubElement(layer, 'Style')
        ET.SubElement(style, 'Name').text = 'generic'
        ET.SubElement(style, 'Title').text = 'Generic'
        ET.SubElement(style, 'Abstract').text = 'Generic Style'

        legend_url = ET.SubElement(style, 'LegendURL',
                                    width='20', height='20')
        ET.SubElement(legend_url, 'Format').text = 'image/png'
        ET.SubElement(legend_url, 'OnlineResource', **{
                'xmlns:xlink': 'http://www.w3.org/1999/xlink',
                'xlink:type': 'simple',
                'xlink:href': ''})




    tree = ET.ElementTree(root)

    return ET.tostring(root, method='xml')
