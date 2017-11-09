import os
import config
import telebot
import classifier
import cv2
import time


#token = 'YOUR TOKEN'
bot = telebot.TeleBot(token)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    files_folder_path = './telegram/'
    ext='.jpg'
    iterator = len(os.listdir(files_folder_path))
    
    raw = message.photo[0].file_id
    file_info = bot.get_file(raw)
    
    bot.send_chat_action(message.chat.id,'typing')
    downloaded_file = bot.download_file(file_info.file_path)
    
    with open(files_folder_path+str(iterator)+ext,'wb') as new_file:
        new_file.write(downloaded_file)
        new_file.close()
    
    img = cv2.imread(files_folder_path+str(iterator)+ext)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print('classifier')
    image_to_send,predicted_class,predicted_probability = classifier.classify(img,iterator)
    
    roi_file_path = files_folder_path+'roi_'+str(iterator)+ext
    
    print('write_roi')
    cv2.imwrite(roi_file_path,image_to_send)
    photo = open(roi_file_path, 'rb')
    new_message_text = 'Type:'+str(predicted_class[0]+1)+'|Probability:%.2f'%(100*(predicted_probability[0][predicted_class[0]]))+'%' 
    
    bot.send_photo(message.chat.id, photo,caption=new_message_text)
    
    os.remove(roi_file_path)


@bot.message_handler(content_types=['document'])
def handle_document(message):
    start_time = time.time()
    files_folder_path = './telegram/'
    ext='.jpg'
    iterator = len(os.listdir(files_folder_path))
    
    raw = message.document.file_id
    file_original_name = message.document.file_name
    file_info = bot.get_file(raw)
    
    bot.send_chat_action(message.chat.id, 'upload_document')
    downloaded_file = bot.download_file(file_info.file_path)
    
    with open(files_folder_path+str(iterator)+ext,'wb') as new_file:
        new_file.write(downloaded_file)
        new_file.close()
    
    img = cv2.imread(files_folder_path+str(iterator)+ext)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print('classifier')
    image_to_send,predicted_class,predicted_probability = classifier.classify(img,iterator)
    
    roi_file_path = files_folder_path+'roi_'+str(file_original_name)
    
    print('write_roi')
    cv2.imwrite(roi_file_path,image_to_send)
    photo = open(roi_file_path, 'rb')
    new_message_text = 'Type: '+str(predicted_class[0]+1)+' | probability: %.2f'%(100*(predicted_probability[0][predicted_class[0]]))+'%' 
    
    bot.send_photo(message.chat.id, photo,caption=new_message_text)
    os.remove(roi_file_path)

    print("Run time: "+str(time.time()-start_time))
    
    
if __name__ == '__main__':
    bot.polling(none_stop=True)

